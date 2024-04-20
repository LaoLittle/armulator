use std::cell::RefCell;
use std::mem::{align_of, offset_of, size_of};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::AddressInstDecoder;
use cranelift::codegen::ir::UserFuncName;
use cranelift::codegen::settings;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, DataDescription, DataId, Module};
use dashmap::DashMap;
use smallvec::SmallVec;
use yaxpeax_arm::armv8;
use yaxpeax_arm::armv8::a64::Opcode::*;
use yaxpeax_arm::armv8::a64::{Operand, ShiftStyle, SizeCode};

use crate::context::{CpuContext, ExecutionReturn, InterruptType, PState};
use crate::error::Error;

use crate::inst::{Condition, Register};

pub struct Translator {
    pub module: JITModule,
    pub signature: Signature,
    pub counter: AtomicU32,
    current_el: u8,
    debug: bool,
}

// (Function ret) (CpuContext* ctx, void* data)
pub type BlockAbi = unsafe extern "C" fn(*mut CpuContext) -> *mut usize;

impl Translator {
    pub fn new() -> crate::error::Result<Self> {
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("use_colocated_libcalls", "false");
        let _ = flag_builder.set("is_pic", "false");
        let _ = flag_builder.set("opt_level", "speed");
        let isa_builder = cranelift_native::builder().map_err(|_| Error::NotSupported)?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|_| Error::NotSupported)?;

        let jb = JITBuilder::with_isa(isa.clone(), default_libcall_names());

        let module = JITModule::new(jb);
        let ptr = module.target_config().pointer_type();
        let abi_ptr = AbiParam::new(ptr);

        let mut signature = module.make_signature();
        signature.params.extend([abi_ptr]);
        signature.returns.push(AbiParam::new(types::I64));

        Ok(Self {
            module,
            signature,
            counter: AtomicU32::new(0),
            current_el: 0,
            debug: false,
        })
    }

    #[inline]
    pub fn set_el(&mut self, el: u8) {
        self.current_el = el;
    }

    pub fn translate<I: Iterator<Item = armv8::a64::Instruction>>(&mut self, insts: I) -> BlockAbi {
        let mut mcx = self.module.make_context();
        mcx.set_disasm(self.debug);

        let block = self
            .module
            .declare_anonymous_function(&self.signature)
            .unwrap();

        mcx.func.signature = self.signature.clone();
        mcx.func.name = UserFuncName::user(0, block.as_u32());

        let mut func_ctx = FunctionBuilderContext::new();
        let mut bcx: FunctionBuilder = FunctionBuilder::new(&mut mcx.func, &mut func_ctx);

        let entry = bcx.create_block();
        let start = bcx.create_block();

        bcx.switch_to_block(entry);
        bcx.append_block_params_for_function_params(entry);

        let cpu_ctx = bcx.block_params(entry)[0];

        bcx.ins().jump(start, &[]);
        bcx.switch_to_block(start);

        let mut tctx = TranslationContext::new(self, bcx, cpu_ctx);

        tctx.translate(insts);
        tctx.finalize();

        let (static_data, dynamic_data) = (tctx.static_data, tctx.dynamic_data);

        let mut bcx = tctx.builder;
        bcx.seal_all_blocks();
        bcx.finalize();

        let mut desc_static = DataDescription::new();
        desc_static.define_zeroinit(size_of::<usize>());
        desc_static.set_align(align_of::<usize>() as u64);

        for id in static_data {
            self.module.define_data(id, &desc_static).unwrap();
        }

        if let Some(id) = dynamic_data {
            let mut desc_dynamic = DataDescription::new();

            desc_dynamic.define_zeroinit(size_of::<DynCache>());
            desc_dynamic.set_align(align_of::<DynCache>() as u64);

            self.module.define_data(id, &desc_dynamic).unwrap();
        }

        self.module.define_function(block, &mut mcx).unwrap();

        self.module.finalize_definitions().unwrap();

        if let Some(vcode) = &mcx.compiled_code().unwrap().vcode {
            println!("{vcode}");
        }

        let func_ptr = self.module.get_finalized_function(block);

        unsafe { std::mem::transmute(func_ptr) }
    }

    #[inline]
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LoadState {
    Loaded,
    Stored,
}

impl LoadState {
    pub fn stored(&self) -> bool {
        matches!(self, Self::Stored)
    }
}

pub struct TranslationContext<'a> {
    translator: &'a mut Translator,
    builder: FunctionBuilder<'a>,
    static_data: SmallVec<DataId, 2>,
    dynamic_data: Option<DataId>,
    cpu_context: Value,
    pc_loadstate: Option<LoadState>,
    gprs_loaded: [Option<LoadState>; 32],
    nzcv_loaded: [Option<LoadState>; 4],

    dynamic_block: Block,

    exec: ExecutionReturn,
}

const PC_VAR: u32 = 50;
const PSTATE_VAR_BASE: u32 = 40;

const PSTATE_N: u32 = 0;
const PSTATE_Z: u32 = 1;
const PSTATE_C: u32 = 2;
const PSTATE_V: u32 = 3;

impl<'a> TranslationContext<'a> {
    #[inline]
    pub fn new(
        translator: &'a mut Translator,
        mut builder: FunctionBuilder<'a>,
        cpu_context: Value,
    ) -> Self {
        // 0..32
        for i in 0..32 {
            builder.declare_var(Variable::from_u32(i), types::I64);
        }

        // 40..44
        for i in 0..4 {
            builder.declare_var(Variable::from_u32(PSTATE_VAR_BASE + i), types::I8);
        }

        // 50
        builder.declare_var(Variable::from_u32(PC_VAR), types::I64);

        let dynamic_block = builder.create_block();

        Self {
            translator,
            builder,
            static_data: SmallVec::new(),
            dynamic_data: None,
            cpu_context,
            pc_loadstate: None,
            gprs_loaded: [None; 32],
            nzcv_loaded: [None; 4],

            dynamic_block,

            exec: ExecutionReturn::new(),
        }
    }

    pub fn translate<I: Iterator<Item = armv8::a64::Instruction>>(&mut self, insts: I) {
        for inst in insts {
            if self.translator.debug {
                eprintln!("{inst}");
            }

            let op = inst.opcode;

            match op {
                ADR | ADRP => {
                    let [rd, label, ..] = inst.operands;

                    let label = Self::get_pcoffset(label);
                    let mut pc = self.get_pc();

                    if matches!(op, ADRP) {
                        pc = self.ins().band_imm(pc, !4095);
                    }

                    let result = self.ins().iadd_imm(pc, label);
                    self.set_operand(rd, result);
                }
                ADD | ADDS => {
                    let [rd, rn, op2, ..] = inst.operands;
                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = self.ins().iadd(op1, op2);

                    self.set_operand(rd, result);

                    if matches!(op, ADDS) {
                        let carry = self.ins().iconst(types::I64, 1);
                        self.check_alu_cond_with_carry(
                            result,
                            op1,
                            op2,
                            carry,
                            Self::get_sizecode(rd),
                            |s, us, x, y| {
                                if us {
                                    s.ins().uadd_overflow(x, y).1
                                } else {
                                    s.ins().sadd_overflow(x, y).1
                                }
                            },
                        );
                    }
                }
                SUB | SUBS => {
                    let [rd, rn, op2, ..] = inst.operands;
                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = self.ins().isub(op1, op2);

                    self.set_operand(rd, result);

                    if matches!(op, SUBS) {
                        let op2 = self.ins().bnot(op2);
                        let carry = self.ins().iconst(types::I64, 1);
                        self.check_alu_cond_with_carry(
                            result,
                            op1,
                            op2,
                            carry,
                            Self::get_sizecode(rd),
                            |s, us, x, y| {
                                if us {
                                    s.ins().uadd_overflow(x, y).1
                                } else {
                                    s.ins().sadd_overflow(x, y).1
                                }
                            },
                        );
                    }
                }
                AND | ANDS => {
                    let [rd, rn, op2, ..] = inst.operands;

                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = self.ins().band(op1, op2);

                    if matches!(op, ANDS) {
                        self.set_neg_zero(result, Self::get_sizecode(rd));
                    }

                    self.set_operand(rd, result);
                }
                BIC | BICS => {
                    let [rd, rn, op2, ..] = inst.operands;

                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = self.ins().band_not(op1, op2);

                    if matches!(op, BICS) {
                        self.set_neg_zero(result, Self::get_sizecode(rd));
                    }

                    self.set_operand(rd, result);
                }
                EOR | EON => {
                    let [rd, rn, op2, ..] = inst.operands;

                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = match op {
                        EOR => self.ins().bxor(op1, op2),
                        EON => self.ins().bxor_not(op1, op2),
                        _ => unreachable!(),
                    };

                    self.set_operand(rd, result);
                }
                ORR | ORN => {
                    let [rd, rn, op2, ..] = inst.operands;

                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(op2);

                    let result = match op {
                        ORR => self.ins().bor(op1, op2),
                        ORN => self.ins().bor_not(op1, op2),
                        _ => unreachable!(),
                    };

                    self.set_operand(rd, result);
                }
                STR => {
                    let [rt, addr, ..] = inst.operands;

                    let val = self.get_operand(rt);
                    let addr = self.get_operand(addr);

                    if Self::is_wreg(rt) {
                        self.ins().istore32(MemFlags::trusted(), val, addr, 0);
                    } else {
                        self.ins().store(MemFlags::trusted(), val, addr, 0);
                    }
                }
                LDR => {
                    let [rt, addr, ..] = inst.operands;

                    let addr = self.get_operand(addr);

                    let val = if Self::is_wreg(rt) {
                        let val = self.ins().load(types::I32, MemFlags::trusted(), addr, 0);

                        self.ins().uextend(types::I64, val)
                    } else {
                        self.ins().load(types::I64, MemFlags::trusted(), addr, 0)
                    };

                    self.set_operand(rt, val);
                }
                MOVZ | MOVN | MOVK => {
                    let [rd, imm, ..] = inst.operands;

                    let Operand::ImmShift(imm, shift) = imm else {
                        unreachable!();
                    };

                    let mut result = (imm as u64) << shift;

                    if matches!(op, MOVN) {
                        result = !result;
                    }

                    let mut result = self.ins().iconst(types::I64, result as i64);

                    if matches!(op, MOVK) {
                        let mask = !((u16::MAX as u64) << shift);
                        let rd = self.get_operand(rd);
                        let masked = self.ins().band_imm(rd, mask as i64);

                        result = self.ins().bor(masked, result);
                    }

                    self.set_operand(rd, result);
                }
                B | BL => {
                    if matches!(op, BL) {
                        let pc = self.get_pc();
                        let lr = self.builder.ins().iadd_imm(pc, 4);
                        self.set_gpr(Register::LR, lr);
                    }

                    let [label, ..] = inst.operands;
                    let label = Self::get_pcoffset(label);

                    self.jump_static(label);
                    return;
                }
                BR | RET | BLR => {
                    if matches!(op, BLR) {
                        let pc = self.get_pc();
                        let lr = self.builder.ins().iadd_imm(pc, 4);
                        self.set_gpr(Register::LR, lr);
                    }

                    let [rn, ..] = inst.operands;
                    let target = self.get_operand(rn);
                    self.jump_dynamic(target);

                    return;
                }
                Bcc(cond) => {
                    let [label, ..] = inst.operands;

                    let label = Self::get_pcoffset(label);

                    let cond = Condition::from_code(cond as u32);
                    let cond = self.load_cond(cond);

                    let tb = self.create_block();
                    let fb = self.create_block();

                    self.ins().brif(cond, tb, &[], fb, &[]);
                    {
                        self.switch_to_block(tb);

                        // return here.
                        self.jump_static(label);
                    }

                    self.switch_to_block(fb);
                }
                CBNZ | CBZ => {
                    let [rt, label, ..] = inst.operands;

                    let op1 = self.get_operand(rt);
                    let label = Self::get_pcoffset(label);

                    let cond = match op {
                        CBNZ => self.check_not_zero(op1),
                        CBZ => self.check_zero(op1),
                        _ => unreachable!(),
                    };

                    let tb = self.builder.create_block();
                    let fb = self.builder.create_block();

                    self.builder.ins().brif(cond, tb, &[], fb, &[]);
                    {
                        self.builder.switch_to_block(tb);

                        // return here.
                        self.jump_static(label);
                    }

                    self.builder.switch_to_block(fb);
                }
                TBNZ | TBZ => {
                    let [rt, bit_pos, offset, ..] = inst.operands;

                    let offset = Self::get_pcoffset(offset);

                    let bit_pos = Self::get_imm16(bit_pos);
                    let mask = 0b1 << bit_pos;
                    let rt = self.get_operand(rt);

                    let bit = self.ins().band_imm(rt, mask);

                    let cond = if matches!(op, TBZ) {
                        self.check_zero(bit)
                    } else {
                        self.check_not_zero(bit)
                    };

                    let tb = self.builder.create_block();
                    let fb = self.builder.create_block();

                    self.builder.ins().brif(cond, tb, &[], fb, &[]);
                    {
                        self.builder.switch_to_block(tb);

                        // return here.
                        self.jump_static(offset);
                    }

                    self.builder.switch_to_block(fb);
                }
                CSEL | CSINC | CSINV | CSNEG => {
                    let [rd, rn, rm, cond] = inst.operands;

                    let op1 = self.get_operand(rn);
                    let op2 = self.get_operand(rm);

                    let op2 = match op {
                        CSEL => op2,
                        CSINC => self.builder.ins().iadd_imm(op2, 1),
                        CSINV => self.builder.ins().bnot(op2),
                        CSNEG => self.builder.ins().ineg(op2),
                        _ => unreachable!(),
                    };

                    let cond = self.get_condition_result(cond);

                    let result = self.builder.ins().select(cond, op1, op2);

                    self.set_operand(rd, result);
                }
                SVC => {
                    let imm = Self::get_imm16(inst.operands[0]);

                    self.exec.set_ty(InterruptType::Svc);
                    self.exec.set_val(imm);

                    break;
                }
                _ => unimplemented!("instruction: {inst}"),
            }

            if self.current_el() >= 1 {
                match op {
                    MRS => unimplemented!(),
                    MSR => unimplemented!(),
                    _ => {}
                }
            }

            self.update_pc();
        }

        self.jump_static(0);
    }

    pub fn finalize(&mut self) {
        if let Some(tc) = self.dynamic_data {
            self.builder.switch_to_block(self.dynamic_block);
            let tc = self
                .translator
                .module
                .declare_data_in_func(tc, &mut self.builder.func);

            let tc_ptr = self.builder.ins().global_value(types::I64, tc);
            let lookup = self
                .builder
                .ins()
                .iconst(types::I64, lookup_and_jump_dynamic as i64);
            let mut sig = self.translator.module.make_signature();
            let abi_ptr = AbiParam::new(types::I64);
            sig.params.extend([abi_ptr, abi_ptr]);

            let sig = self.builder.import_signature(sig);

            self.builder
                .ins()
                .call_indirect(sig, lookup, &[self.cpu_context, tc_ptr]);

            let ret = self
                .builder
                .ins()
                .iadd_imm(self.cpu_context, offset_of!(CpuContext, next_block) as i64);

            self.builder.ins().return_(&[ret]);
        }
    }

    pub fn jump_static(&mut self, label: i64) {
        let pc = self.get_pc();
        let target = self.builder.ins().iadd_imm(pc, label);

        self.set_pc(target);
        self.save_context();

        // for each branch, we add a new pointer to the data section,
        // so it would be isolated,
        // and we don't have to consider the risk of data racing
        let tc = self
            .translator
            .module
            .declare_anonymous_data(true, false)
            .unwrap();
        self.static_data.push(tc);
        let tc = self
            .translator
            .module
            .declare_data_in_func(tc, &mut self.builder.func);

        let tc_ptr = self.builder.ins().global_value(types::I64, tc);

        self.builder.ins().return_(&[tc_ptr]);
    }

    pub fn jump_dynamic(&mut self, dest: Value) {
        self.dynamic_data.get_or_insert_with(|| {
            self.translator
                .module
                .declare_anonymous_data(true, false)
                .unwrap()
        });

        self.set_pc(dest);
        self.save_context();

        self.builder.ins().jump(self.dynamic_block, &[]);
    }

    #[inline]
    pub const fn current_el(&self) -> u8 {
        self.translator.current_el
    }

    pub fn is_xreg(operand: Operand) -> bool {
        matches!(operand, Operand::Register(SizeCode::X, ..))
    }

    pub fn is_wreg(operand: Operand) -> bool {
        !Self::is_xreg(operand)
    }

    pub fn get_operand(&mut self, operand: Operand) -> Value {
        match operand {
            Operand::ImmShift(imm, shift) => {
                let result = (imm as u64) << shift;

                self.builder.ins().iconst(types::I64, result as i64)
            }
            Operand::Immediate(imm) => self.ins().iconst(types::I64, imm as i64),
            Operand::Register(c, id) => {
                let mut gpr = self.get_gpr(Register::new_with_zr(id as u8));

                if matches!(c, SizeCode::W) {
                    gpr = self.normalize32(gpr);
                }

                gpr
            }
            Operand::RegShift(s, shift, c, rn) => {
                let mut rn = self.get_gpr(Register::new_with_zr(rn as u8));

                if matches!(c, SizeCode::W) {
                    rn = self.normalize32(rn);
                }

                match s {
                    ShiftStyle::LSL => self.ins().ishl_imm(rn, shift as i64),
                    ShiftStyle::LSR => self.ins().ushr_imm(rn, shift as i64),
                    ShiftStyle::ASR => self.ins().sshr_imm(rn, shift as i64),
                    ShiftStyle::ROR => self.ins().rotr_imm(rn, shift as i64),
                    _ => unimplemented!(),
                }
            }
            Operand::RegisterOrSP(c, id) => {
                let mut gpr = self.get_gpr(Register::new_with_sp(id as u8));

                if matches!(c, SizeCode::W) {
                    gpr = self.normalize32(gpr);
                }

                gpr
            }
            Operand::RegPreIndex(rn, imm, wb) => {
                let reg = Register::new_with_sp(rn as u8);
                let rn = self.get_gpr(reg);
                let addr = self.ins().iadd_imm(rn, imm as i64);

                if wb {
                    self.set_gpr(reg, addr);
                }

                addr
            }
            Operand::PCOffset(label) => {
                let pc = self.get_pc();
                self.builder.ins().iadd_imm(pc, label)
            }
            _ => unimplemented!("Operand: {operand:?}"),
        }
    }

    pub fn set_operand(&mut self, operand: Operand, mut value: Value) {
        match operand {
            Operand::Register(c, id) => {
                if matches!(c, SizeCode::W) {
                    value = self.normalize32(value);
                }

                self.set_gpr(Register::new_with_zr(id as u8), value);
            }
            Operand::RegisterOrSP(c, id) => {
                if matches!(c, SizeCode::W) {
                    value = self.normalize32(value);
                }

                self.set_gpr(Register::new_with_sp(id as u8), value);
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn get_pcoffset(operand: Operand) -> i64 {
        match operand {
            Operand::PCOffset(label) => label,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn get_imm16(operand: Operand) -> u16 {
        match operand {
            Operand::Imm16(imm) => imm,
            _ => unreachable!(),
        }
    }

    pub fn get_sizecode(operand: Operand) -> SizeCode {
        match operand {
            Operand::Register(s, ..) => s,
            _ => unimplemented!(),
        }
    }

    pub fn get_gpr(&mut self, rn: Register) -> Value {
        if rn.is_zero() {
            return self.ins().iconst(types::I64, 0);
        }
        let idx = rn.index();
        let var = Variable::from_u32(idx as u32);

        if self.gprs_loaded[idx as usize].is_none() {
            let loaded = self.builder.ins().load(
                types::I64,
                MemFlags::trusted(),
                self.cpu_context,
                (offset_of!(CpuContext, gprs) + size_of::<u64>() * idx as usize) as i32,
            );

            self.builder.def_var(var, loaded);
            self.gprs_loaded[idx as usize] = Some(LoadState::Loaded);
        }

        self.builder.use_var(var)
    }

    pub fn set_gpr(&mut self, rn: Register, new: Value) {
        if rn.is_zero() {
            return;
        }

        let idx = rn.index();

        self.builder.def_var(Variable::from_u32(idx as u32), new);
        self.gprs_loaded[idx as usize] = Some(LoadState::Stored);
    }

    pub fn get_pstate(&mut self, select: u32) -> Value {
        let var = Variable::from_u32(PSTATE_VAR_BASE + select);
        if self.nzcv_loaded[select as usize].is_none() {
            let loaded = self.builder.ins().load(
                types::I8,
                MemFlags::trusted(),
                self.cpu_context,
                (offset_of!(CpuContext, pstate)
                    + offset_of!(PState, condition_flags)
                    + size_of::<u8>() * select as usize) as i32,
            );

            self.builder.def_var(var, loaded);
            self.nzcv_loaded[select as usize] = Some(LoadState::Loaded);
        }

        self.builder.use_var(var)
    }

    pub fn get_n(&mut self) -> Value {
        self.get_pstate(PSTATE_N)
    }

    pub fn get_z(&mut self) -> Value {
        self.get_pstate(PSTATE_Z)
    }

    pub fn get_c(&mut self) -> Value {
        self.get_pstate(PSTATE_C)
    }

    pub fn get_v(&mut self) -> Value {
        self.get_pstate(PSTATE_V)
    }

    pub fn set_pstate(&mut self, select: u32, val: Value) {
        let val = utils::i2bool(&mut self.builder, val);

        self.builder
            .def_var(Variable::from_u32(PSTATE_VAR_BASE + select), val);
        self.nzcv_loaded[select as usize] = Some(LoadState::Stored);
    }

    pub fn set_n(&mut self, val: Value) {
        self.set_pstate(PSTATE_N, val);
    }

    pub fn set_z(&mut self, val: Value) {
        self.set_pstate(PSTATE_Z, val);
    }

    pub fn set_c(&mut self, val: Value) {
        self.set_pstate(PSTATE_C, val);
    }

    pub fn set_v(&mut self, val: Value) {
        self.set_pstate(PSTATE_V, val);
    }

    pub fn normalize32(&mut self, value: Value) -> Value {
        let reduced = self.builder.ins().ireduce(types::I32, value);
        self.builder.ins().uextend(types::I64, reduced)
    }

    pub fn get_pc(&mut self) -> Value {
        if self.pc_loadstate.is_none() {
            // load the pc from memory
            let loaded = self.builder.ins().load(
                types::I64,
                MemFlags::trusted(),
                self.cpu_context,
                offset_of!(CpuContext, pc) as i32,
            );

            self.builder.def_var(Variable::from_u32(PC_VAR), loaded);
            self.pc_loadstate = Some(LoadState::Loaded);
        }

        self.builder.use_var(Variable::from_u32(PC_VAR))
    }

    pub fn set_pc(&mut self, new: Value) {
        self.builder.def_var(Variable::from_u32(PC_VAR), new);
        self.pc_loadstate = Some(LoadState::Stored);
    }

    pub fn get_condition_result(&mut self, operand: Operand) -> Value {
        match operand {
            Operand::ConditionCode(code) => self.load_cond(Condition::from_code(code as u32)),
            _ => unreachable!(),
        }
    }

    pub fn load_cond(&mut self, cond: Condition) -> Value {
        use Condition::*;

        match cond {
            Eq => self.get_z(),
            Ne => {
                let cond = self.get_z();
                self.check_zero(cond)
            }
            Cs => self.get_c(),
            Cc => {
                let cond = self.get_c();
                self.check_zero(cond)
            }
            Mi => self.get_n(),
            Pl => {
                let cond = self.get_n();
                self.check_zero(cond)
            }
            Vs => self.get_v(),
            Vc => {
                let cond = self.get_v();
                self.check_zero(cond)
            }
            Hi => {
                let c = self.get_c();
                let z = self.get_z();
                let z = self.check_zero(z);
                self.builder.ins().band(c, z)
            }
            Ls => {
                let cond = {
                    let c = self.get_c();
                    let z = self.get_z();
                    let z = self.check_zero(z);
                    self.builder.ins().band(c, z)
                };

                self.check_zero(cond)
            }
            Ge => {
                let n = self.get_n();
                let v = self.get_v();

                utils::is_eq(&mut self.builder, n, v)
            }
            Lt => {
                let n = self.get_n();
                let v = self.get_v();

                utils::is_neq(&mut self.builder, n, v)
            }
            Gt => {
                let c0 = {
                    let n = self.get_n();
                    let v = self.get_v();

                    utils::is_eq(&mut self.builder, n, v)
                };

                let z = self.get_z();
                let z = self.check_zero(z);

                self.ins().band(c0, z)
            }
            Le => {
                let cond = {
                    let c0 = {
                        let n = self.get_n();
                        let v = self.get_v();

                        utils::is_eq(&mut self.builder, n, v)
                    };

                    let z = self.get_z();
                    let z = self.check_zero(z);

                    self.builder.ins().band(c0, z)
                };

                self.check_zero(cond)
            }
            Al | Nv => self.builder.ins().iconst(types::I8, 1),
        }
    }

    pub fn check_alu_cond_with_carry(
        &mut self,
        val: Value,
        a: Value,
        b: Value,
        cin: Value,
        sc: SizeCode,
        mut checker: impl FnMut(&mut Self, bool, Value, Value) -> Value,
    ) {
        // todo: optimize it
        // true if the value is negative
        self.set_neg_zero(val, sc);

        let ab = self.ins().iadd(a, b);
        let c0 = checker(self, true, a, b);
        let v0 = checker(self, false, a, b);

        let c1 = checker(self, true, ab, cin);
        let v1 = checker(self, false, ab, cin);

        let c = self.ins().bor(c0, c1);
        let v = self.ins().bor(v0, v1);

        self.set_c(c);
        self.set_v(v);
    }

    pub fn update_pc(&mut self) {
        let pc = self.get_pc();
        let updated = self.builder.ins().iadd_imm(pc, 4);
        self.set_pc(updated);
    }

    pub fn save_context(&mut self) {
        let pc = self.get_pc();

        self.builder.ins().store(
            MemFlags::trusted(),
            pc,
            self.cpu_context,
            offset_of!(CpuContext, pc) as i32,
        );

        for (id, state) in self.gprs_loaded.into_iter().enumerate() {
            if !matches!(state, Some(LoadState::Stored)) {
                continue;
            }

            let reg = self.get_gpr(Register::new_with_sp(id as u8));
            self.builder.ins().store(
                MemFlags::trusted(),
                reg,
                self.cpu_context,
                (offset_of!(CpuContext, gprs) + size_of::<u64>() * id) as i32,
            );
        }

        for (id, state) in self.nzcv_loaded.into_iter().enumerate() {
            if !matches!(state, Some(LoadState::Stored)) {
                continue;
            }

            let val = self.get_pstate(id as u32);
            self.builder.ins().store(
                MemFlags::trusted(),
                val,
                self.cpu_context,
                (offset_of!(CpuContext, pstate)
                    + offset_of!(PState, condition_flags)
                    + size_of::<u8>() * id) as i32,
            );
        }

        if !matches!(self.exec.ty(), InterruptType::None) || self.exec.val() != 0 {
            let exec = self
                .builder
                .ins()
                .iconst(types::I32, self.exec.into_u32() as i64);
            self.builder.ins().store(
                MemFlags::trusted(),
                exec,
                self.cpu_context,
                offset_of!(CpuContext, status) as i32,
            );
        }
    }

    pub fn check_neg(&mut self, val: Value, size_code: SizeCode) -> Value {
        let val = if matches!(size_code, SizeCode::X) {
            val
        } else {
            self.ins().ireduce(types::I32, val)
        };

        self.ins().icmp_imm(IntCC::SignedLessThan, val, 0)
    }

    pub fn check_zero(&mut self, val: Value) -> Value {
        self.ins().icmp_imm(IntCC::Equal, val, 0)
    }

    pub fn check_not_zero(&mut self, val: Value) -> Value {
        self.ins().icmp_imm(IntCC::NotEqual, val, 0)
    }

    pub fn set_neg_zero(&mut self, val: Value, sc: SizeCode) {
        let n = self.check_neg(val, sc);
        self.set_n(n);
        let z = self.check_zero(val);
        self.set_z(z);
    }
}

impl<'a> Deref for TranslationContext<'a> {
    type Target = FunctionBuilder<'a>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<'a> DerefMut for TranslationContext<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

pub fn translation_cache() -> &'static DashMap<u64, BlockAbi> {
    static CACHE: OnceLock<DashMap<u64, BlockAbi>> = OnceLock::new();

    CACHE.get_or_init(|| DashMap::new())
}

thread_local! {
    static TRANSLATOR: RefCell<Translator> = RefCell::new(Translator::new().unwrap())
}

// At first, the data equals the address of this function.
// Then, we replace the data provided the lookup or translation succeeded,
// so this function would not be called by the same block twice,
// and the block would jump to the target directly.
/*unsafe extern "C" fn lookup_and_jump_static(ctx: *mut CpuContext, data: *mut usize) -> *mut usize {
    let fnptr = *data;
    if fnptr != lookup_and_jump_static as usize {
        return data;
    }

    let target = (*ctx).pc;
    // lookup from global cache or translate here..

    // *data = translated func;
    // finally, we call the function and return
    // block(ctx, data)




}*/

type DynCache = [[u64; 2]; 4];

unsafe extern "C" fn lookup_and_jump_dynamic(ctx: *mut CpuContext, data: *mut DynCache) {
    static COUNTER: AtomicU8 = AtomicU8::new(0);

    let ctx = &mut *ctx;
    let data = &mut *data;

    let target = ctx.pc;

    if target == 0 {
        let mut s = ExecutionReturn::new();
        s.set_ty(InterruptType::Udf);
        ctx.status = s.into_u32();
        return;
    }

    // The data contains a pointer to an array
    // which contains the mappings from guest pc to compiled block
    // lookup the local cache here.
    for [k, v] in *data {
        if k == target {
            if v == 0 {
                eprintln!("Cache hits the target pointer: 0x{k:x}, while the compiled code is null. Abort..");
                std::process::abort();
            }
            ctx.next_block = v as usize;

            return;
        }
    }

    let gc = translation_cache();

    // lookup the global cache or translate here.
    let block = 'lookup: {
        if let Some(cached) = gc.get(&target) {
            break 'lookup *cached;
        }

        let compiled =
            TRANSLATOR.with_borrow_mut(|t| t.translate(AddressInstDecoder::new(target as usize)));
        gc.insert(target, compiled);

        compiled
    } as usize;

    let idx = (COUNTER.fetch_add(1, Ordering::Relaxed) & 3) as usize;

    (*data)[idx] = [target, block as u64];

    ctx.next_block = block;
}

mod utils {
    use cranelift::prelude::*;

    /// cast any integer type into a bool value
    pub fn i2bool(bcx: &mut FunctionBuilder, val: Value) -> Value {
        // val != 0
        bcx.ins().icmp_imm(IntCC::NotEqual, val, 0)
    }

    pub fn is_zero(bcx: &mut FunctionBuilder, val: Value) -> Value {
        // val == 0
        bcx.ins().icmp_imm(IntCC::Equal, val, 0)
    }

    pub fn is_eq(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        bcx.ins().icmp(IntCC::Equal, a, b)
    }

    pub fn is_neq(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        bcx.ins().icmp(IntCC::NotEqual, a, b)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::transmute;
    use std::ptr::null_mut;

    use yaxpeax_arm::armv8::a64::{Instruction, Opcode, Operand, SizeCode};

    use crate::context::CpuContext;
    use crate::translate_id::{BlockAbi, Translator, TRANSLATOR};

    #[test]
    fn static_jump() {
        let mut translator = Translator::new().unwrap();

        translator.debug = true;

        let ptr = TRANSLATOR.with_borrow_mut(|t| {
            t.debug = true;
            t.translate(
                [
                    Instruction {
                        opcode: Opcode::TBZ,
                        operands: [
                            Operand::Register(SizeCode::X, 0),
                            Operand::Imm16(1),
                            Operand::PCOffset(114),
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::MOVZ,
                        operands: [
                            Operand::Register(SizeCode::X, 1),
                            Operand::ImmShift(114, 0),
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::MOVZ,
                        operands: [
                            Operand::Register(SizeCode::X, 2),
                            Operand::ImmShift(514, 0),
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::CSINC,
                        operands: [
                            Operand::Register(SizeCode::X, 4),
                            Operand::Register(SizeCode::X, 1),
                            Operand::Register(SizeCode::X, 6),
                            Operand::ConditionCode(0), // eq
                        ],
                    },
                    Instruction {
                        opcode: Opcode::MOVK,
                        operands: [
                            Operand::Register(SizeCode::X, 10),
                            Operand::ImmShift(0b1111, 0),
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::MOVK,
                        operands: [
                            Operand::Register(SizeCode::X, 10),
                            Operand::ImmShift(0b1111, 16),
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::B,
                        operands: [
                            Operand::PCOffset(8),
                            Operand::Nothing,
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                ]
                .into_iter(),
            )
        });

        let mut ctx = CpuContext::new();

        *ctx.gpr_mut(0) = 0b0010;
        ctx.pstate.condition_flags[1] = 0; // z = true

        let mut next = null_mut();
        let mut f = ptr;
        unsafe {
            next = f(&mut ctx);

            println!("STATUS: {:?}", ctx.status());
            println!("PC: {}", ctx.pc());

            assert_eq!(ctx.gprs[1], 114);
            assert_eq!(ctx.gprs[2], 514);

            println!("X4: {}", ctx.gprs[4]);

            println!("X10: {:b}", ctx.gprs[10]);

            assert_eq!(*next, 0);
            panic!();
        }
    }

    #[test]
    fn dynamic() {
        let mut translator = Translator::new().unwrap();

        translator.debug = true;

        let ptr = TRANSLATOR.with_borrow_mut(|t| {
            t.debug = true;
            t.translate(
                [
                    Instruction {
                        opcode: Opcode::MOVZ,
                        operands: [
                            Operand::Register(SizeCode::X, 20),
                            Operand::ImmShift(114, 0),
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    },
                    Instruction {
                        opcode: Opcode::BR,
                        operands: [
                            Operand::Register(SizeCode::X, 20),
                            Operand::Nothing,
                            Operand::Nothing,
                            Operand::Nothing,
                        ],
                    }, /*Inst::Movz {
                           rd: Register::General(20),
                           imm: 114,
                           shift: 0,
                           sf: false,
                       },
                       Inst::Br { rn: 20 },*/
                ]
                .into_iter(),
            )
        });

        let mut ctx = CpuContext::new();

        let mut next = null_mut();
        let mut f = ptr;

        unsafe {
            next = f(&mut ctx);

            assert_eq!(&ctx.next_block as *const usize, next);

            println!("STATUS: {:?}", ctx.status());
            println!("PC: {}", ctx.pc());

            let nnn: BlockAbi = transmute(*next);
            next = nnn(&mut ctx);

            panic!("{}", *next);
        }
    }
}

/*
match inst {
                Movz { rd, imm, shift, sf } | Movn { rd, imm, shift, sf } => {
                    let mut result = (imm as u64) << shift;

                    if matches!(inst, Movn { .. }) {
                        result = !result;
                    }

                    let mut result = self.builder.ins().iconst(types::I64, result as i64);

                    if !sf {
                        result = self.normalize32(result);
                    }

                    self.set_gpr(rd, result);
                }
                B { label } | Bl { label } => {
                    if matches!(inst, Bl { .. }) {
                        let pc = self.get_pc();
                        let lr = self.builder.ins().iadd_imm(pc, 4);
                        self.set_gpr(Register::LR, lr);
                    }

                    self.jump_static(label);
                    return;
                }
                Br { rn } | Ret { rn } => {
                    let target = self.get_gpr(Register::General(rn));
                    self.jump_dynamic(target);
                    return;
                }
                Tbz {
                    rt,
                    bit_pos,
                    offset,
                    sf,
                }
                | Tbnz {
                    rt,
                    bit_pos,
                    offset,
                    sf,
                } => {
                    let mask = 0b1 << bit_pos;
                    let mut rt = self.get_gpr(Register::General(rt));

                    if !sf {
                        rt = self.normalize32(rt);
                    }

                    let bit = self.builder.ins().band_imm(rt, mask);

                    let cond = if matches!(inst, Tbz { .. }) {
                        utils::is_zero(&mut self.builder, bit)
                    } else {
                        utils::i2bool(&mut self.builder, bit)
                    };

                    let tb = self.builder.create_block();
                    let fb = self.builder.create_block();

                    self.builder.ins().brif(cond, tb, &[], fb, &[]);
                    {
                        self.builder.switch_to_block(tb);

                        // return here.
                        self.jump_static(offset);
                    }

                    self.builder.switch_to_block(fb);
                    // translate the remaining instructions
                }
                Svc { imm } => {
                    self.exec.set_ty(InterruptType::Svc);
                    self.exec.set_val(imm);

                    break;
                }
                Udf { imm } => {
                    self.exec.set_ty(InterruptType::Udf);
                    self.exec.set_val(imm);

                    break;
                }
                Nop => {
                    self.builder.ins().nop();
                }
                _ => unimplemented!()
            }
*/

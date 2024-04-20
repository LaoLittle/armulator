use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};
use std::ptr::null_mut;

use modular_bitfield::{bitfield, BitfieldSpecifier};
use crate::AddressInstDecoder;


use crate::error::Result;
use crate::translate_id::{BlockAbi, translation_cache, Translator};

pub struct Cpu {
    context: CpuContext,
    translator: Translator,
}

impl Cpu {
    #[inline]
    pub fn new() -> Self {
        Self {
            context: CpuContext::new(),
            translator: Translator::new().unwrap(),
        }
    }

    pub unsafe fn execute_inf(&mut self) -> Result<CpuStatus> {
        let mut next = 0;
        let mut ptr: *mut usize = null_mut();
        
        loop {
            let pc = self.context.pc();
            
            if pc == 0 { break; }
            
            if next == 0 {
                let cache = translation_cache();
                
                let block = if let Some(block) = cache.get(&pc) {
                    *block
                } else {
                    let block = self
                        .translator
                        .translate(AddressInstDecoder::new(self.pc() as usize));
                    
                    cache.insert(self.pc(), block);

                    block
                };

                next = block as usize;

                if !ptr.is_null() {
                    *ptr = next;
                }
            }

            let block: BlockAbi = unsafe { std::mem::transmute(next) };

            ptr = unsafe { block(&mut self.context) };
            next = unsafe { *ptr };
        }

        Ok(CpuStatus::from_u32(self.context.status))
    }

    /*/// S: Steps
    pub unsafe fn execute<const S: usize>(&mut self) -> Result<CpuStatus> {
        let pc = self.context.pc();

        if let Some(&block) = self.code_cache.get(&pc) {
            let status = unsafe { block(&mut self.context) };

            return Ok(CpuStatus::from_u32(status));
        }

        // no compiled code found, let's compile.
        let mut optmizer = Optimizer::new();
        for i in 0..S {
            let addr = (pc as *const u32).add(i);
            let mem = unsafe { *addr };

            let mem = mem.to_le_bytes();

            let mut decoder = crate::inst::InstDecoder::new(mem.as_slice());
            let Ok(inst) = decoder.decode_inst() else {
                if i == 0 {
                    return Err(Error::NotSupported);
                }

                break;
            };

            optmizer.perform(inst);
        }

        let v = optmizer.finalize();

        let mut map = HashMap::new();
        map.insert(0, v.into());
        let block = self.translator.translate_blocks(map);

        let status = unsafe { block(&mut self.context) };

        Ok(CpuStatus::from_u32(status))
    }*/

    #[inline]
    pub const fn pc(&self) -> u64 {
        self.context.pc()
    }

    #[inline]
    pub const fn lr(&self) -> u64 {
        self.context.lr()
    }

    #[inline]
    pub fn set_pc(&mut self, base: u64) {
        self.context.pc = base;
    }

    #[inline]
    pub fn set_sp(&mut self, top: u64) {
        *self.context.gpr_mut(REG_SP) = top;
    }

    pub fn set_lr(&mut self, base: u64) {
        *self.context.lr_mut() = base;
    }

    #[inline]
    pub fn context(&self) -> &CpuContext {
        &self.context
    }

    #[inline]
    pub fn set_debug(&mut self, debug: bool) {
        self.translator.set_debug(debug);
    }
}

impl Debug for Cpu {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cpu")
            .field("context", &self.context)
            .finish()
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct CpuContext {
    // general purpose registers
    pub gprs: [u64; 32],
    // program counter
    pub pc: u64,
    pub status: u32,

    // process state
    pub pstate: PState,

    pub next_block: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct PState {
    pub condition_flags: [u8; 4],
    pub current_el: u8,
}

impl CpuContext {
    #[inline]
    pub fn status(&self) -> CpuStatus {
        CpuStatus::from_u32(self.status)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupt {
    Svc,
    Udf,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStatus {
    pub break_type: Option<Interrupt>,
    pub val: u16,
}

pub enum Trap {
    SystemAccess,
}

impl CpuStatus {
    #[inline]
    pub fn normal() -> Self {
        Self {
            break_type: None,
            val: 0,
        }
    }

    pub fn from_u32(ret: u32) -> Self {
        let exec = ExecutionReturn::from_u32(ret);

        Self::from(exec)
    }
}

impl From<ExecutionReturn> for CpuStatus {
    fn from(ret: ExecutionReturn) -> Self {
        Self {
            break_type: match ret.ty() {
                InterruptType::None => None,
                InterruptType::Udf => Some(Interrupt::Udf),
                InterruptType::Svc => Some(Interrupt::Svc),
            },
            val: ret.val(),
        }
    }
}

// special registers' index
pub const REG_FP: u8 = 29;
pub const REG_LR: u8 = 30;
pub const REG_SP: u8 = 31;

pub const REG_ZR: u8 = 31;

impl CpuContext {
    #[inline]
    pub fn new() -> Self {
        Self {
            gprs: [0; 32],
            pc: 0,
            status: 0,
            pstate: PState {
                condition_flags: [0; 4],
                current_el: 0,
            },
            next_block: 0,
        }
    }

    #[inline]
    pub fn gpr_mut(&mut self, rd: u8) -> &mut u64 {
        assert!(rd <= 31, "destination register must less than 31");
        &mut self.gprs[rd as usize]
    }

    #[inline]
    pub const fn fp(&self) -> u64 {
        self.gprs[29]
    }

    #[inline]
    pub const fn lr(&self) -> u64 {
        self.gprs[30]
    }

    #[inline]
    pub fn lr_mut(&mut self) -> &mut u64 {
        &mut self.gprs[30]
    }

    #[inline]
    pub const fn sp(&self) -> u64 {
        self.gprs[31]
    }

    #[inline]
    pub const fn pc(&self) -> u64 {
        self.pc
    }

    #[inline]
    pub fn pc_mut(&mut self) -> &mut u64 {
        &mut self.pc
    }
}

#[derive(BitfieldSpecifier, Copy, Clone, Debug, PartialEq, Eq)]
#[bits = 16]
pub enum InterruptType {
    None = 0,
    Udf = 1,
    Svc = 2,
}

#[derive(Copy, Clone)]
#[bitfield]
pub struct ExecutionReturn {
    pub ty: InterruptType,
    pub val: u16,
}

impl ExecutionReturn {
    #[inline]
    pub const fn into_u32(self) -> u32 {
        u32::from_ne_bytes(self.into_bytes())
    }

    #[inline]
    pub const fn from_u32(i: u32) -> Self {
        Self::from_bytes(i.to_ne_bytes())
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use crate::context::{Cpu, CpuContext};
    use crate::translate_id::translation_cache;

    #[test]
    fn execute() {
        // Adr { rd: 0, label: 12 }
        // B { label: 4 }
        let inst = [0x60u8, 0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x14];

        let mut cpu = Cpu::new();
        cpu.set_debug(true);

        let pc = inst.as_ptr() as u64;
        cpu.set_pc(pc);
        let status = unsafe { cpu.execute_inf().unwrap() };

        dbg!(status);

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], pc + 12);
        assert_eq!(cpu.context().pc(), pc + 4 + 4);

        cpu.set_pc(pc);
        let status = unsafe { cpu.execute_inf().unwrap() };

        dbg!(status);

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], pc + 12);
        assert_eq!(cpu.context().pc(), pc + 4 + 4);
    }

    #[test]
    fn test_control_flow() {
        // int main() {
        //  int ab = 1;
        //
        //  for (int i = 0; i < 10; i++) {
        //       ab += 1;
        //  }
        //
        //  return ab;
        // }

        const STACK_SIZE: usize = 64;
        let mut stack = Box::new([1u8; STACK_SIZE]);
        let mem = [
            0xff, 0x43, 0x00, 0xd1, 0xff, 0x0f, 0x00, 0xb9, 0x28, 0x00, 0x80, 0x52, 0xe8, 0x0b, 0x00, 0xb9,
            0xff, 0x07, 0x00, 0xb9, 0x01, 0x00, 0x00, 0x14, 0xe8, 0x07, 0x40, 0xb9, 0x09, 0x40, 0x99, 0x52,
            0x49, 0x73, 0xa7, 0x72, 0x08, 0x01, 0x09, 0x6b, 0xe8, 0xb7, 0x9f, 0x1a, 0x48, 0x01, 0x00, 0x37,
            0x01, 0x00, 0x00, 0x14, 0xe8, 0x0b, 0x40, 0xb9, 0x08, 0x05, 0x00, 0x11, 0xe8, 0x0b, 0x00, 0xb9,
            0x01, 0x00, 0x00, 0x14, 0xe8, 0x07, 0x40, 0xb9, 0x08, 0x05, 0x00, 0x11, 0xe8, 0x07, 0x00, 0xb9,
            0xf2, 0xff, 0xff, 0x17, 0xe0, 0x0b, 0x40, 0xb9, 0xff, 0x43, 0x00, 0x91, 0xc0, 0x03, 0x5f, 0xd6u8
        ];

        let mut cpu = Cpu::new();

        cpu.set_debug(true);

        let pc = mem.as_ptr() as u64;
        cpu.set_pc(pc);
        cpu.set_sp(unsafe { stack.as_mut_ptr().byte_add(STACK_SIZE) as u64 });

        *cpu.context.lr_mut() = 0;
        
        let now = SystemTime::now();
        
        
        translation_cache().insert(0x10000, some_block); // register a native block

        let _ = unsafe { cpu.execute_inf().unwrap() };
        
        assert_eq!(cpu.context().gprs[0], 1000000001);
        
        panic!("{}", now.elapsed().unwrap().as_secs_f32());

        drop(stack);
    }

    unsafe extern "C" fn some_block(ctx: *mut CpuContext) -> *mut usize {
        //panic!("HEY");

        &mut (*ctx).next_block
    }
}

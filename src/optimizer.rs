use crate::inst::{
    Inst::{self, *},
    Register,
};
use smallvec::SmallVec;

#[derive(Debug, Copy, Clone)]
pub enum SelectInst {
    Raw(Inst),
    Optimized {
        /// optimized instruction
        inst: OptInst,
        /// the amount of raw instructions
        amount: u8,
    },
}

#[derive(Debug, Copy, Clone)]
pub enum OptInst {
    // to update the program counter
    Nop,
    // moves the imm into the register
    Mov { rd: Register, imm: i64 },
}

pub struct Optimizer {
    pub cache: SmallVec<Inst, 16>,
    pub select: Vec<SelectInst>,
}

impl Optimizer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            cache: SmallVec::new(),
            select: Vec::new(),
        }
    }

    pub fn perform(&mut self, inst: Inst) {
        self.cache.push(inst);

        match self.cache[..] {
            [Nop] => {
                self.select.push(SelectInst::Optimized {
                    inst: OptInst::Nop,
                    amount: 1,
                });
            }
            // SUB x0, x0, x0 == MOV x0, 0
            [Sub { rd, rn, op2, .. }] => {
                if rd != rn || rd != op2 {
                    return;
                }

                self.select.push(SelectInst::Optimized {
                    inst: OptInst::Mov { rd, imm: 0 },
                    amount: 1,
                })
            }
            // MOV x0, a
            // MOVK x0, b, LSL #16
            // MOVK x0, c, LSL #32
            // MOVK x0, d, LSL #48
            // == MOV x0, d:c:b:a
            [Movk {
                rd,
                imm: imm0,
                shift: 0,
                sf: true,
            }
            | Movz {
                rd,
                imm: imm0,
                shift: 0,
                sf: true,
            }, Movk {
                rd: r1,
                imm: imm1,
                shift: 16,
                sf: true,
            }, Movk {
                rd: r2,
                imm: imm2,
                shift: 32,
                sf: true,
            }, Movk {
                rd: r3,
                imm: imm3,
                shift: 48,
                sf: true,
            }] => {
                if rd != r1 || rd != r2 || rd != r3 {
                    return;
                }

                let mut imm = 0u64;

                for i in [imm3, imm2, imm1, imm0] {
                    imm <<= 16;
                    imm += i as u64;
                }

                self.select.push(SelectInst::Optimized {
                    inst: OptInst::Mov {
                        rd,
                        imm: imm as i64,
                    },
                    amount: 4,
                });
            }
            // MOV w0, a
            // MOVK w0, b, LSL #16
            // == MOV w0, b:a
            [Movk {
                rd,
                imm: imm0,
                shift: 0,
                sf: false,
            }
            | Movz {
                rd,
                imm: imm0,
                shift: 0,
                sf: false,
            }, Movk {
                rd: r1,
                imm: imm1,
                shift: 16,
                sf: false,
            }] => {
                if rd != r1 {
                    return;
                }

                let mut imm = 0u32;

                for i in [imm1, imm0] {
                    imm <<= 16;
                    imm += i as u32;
                }

                self.select.push(SelectInst::Optimized {
                    inst: OptInst::Mov {
                        rd,
                        imm: imm as i64,
                    },
                    amount: 2,
                });
            }
            _ => return,
        }

        self.cache.clear();
    }

    pub fn finalize(mut self) -> Vec<SelectInst> {
        for inst in self.cache {
            self.select.push(SelectInst::Raw(inst));
        }

        self.select
    }
}

#[cfg(test)]
mod tests {
    use crate::inst::Inst::{Movk, Movz};
    use crate::inst::Register;
    use crate::optimizer::{OptInst, Optimizer, SelectInst};

    #[test]
    fn test_mov() {
        let mut opt = Optimizer::new();

        let rd = Register::new_with_zr(0);
        for inst in [
            Movz {
                rd,
                imm: 60159,
                shift: 0,
                sf: true,
            },
            Movk {
                rd,
                imm: 65279,
                shift: 16,
                sf: true,
            },
            Movk {
                rd,
                imm: 65533,
                shift: 32,
                sf: true,
            },
            Movk {
                rd,
                imm: 65530,
                shift: 48,
                sf: true,
            },
        ] {
            opt.perform(inst);
        }

        let opt = opt.finalize();

        assert!(matches!(
            opt[..],
            [SelectInst::Optimized {
                inst: OptInst::Mov {
                    rd: Register::General(0),
                    imm: -1407383490270465
                },
                amount: 4
            }]
        ));
    }
}

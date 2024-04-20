use smallvec::SmallVec;
use std::collections::{BTreeSet, HashMap};

use crate::inst::Inst;
use crate::optimizer::SelectInst;

pub type BasicBlockMap = HashMap<i64, SmallVec<SelectInst, 32>>;

#[derive(Debug, Clone)]
pub struct BranchAnalyzer {
    pub targets: BTreeSet<u64>,
}

impl BranchAnalyzer {
    #[inline]
    pub fn new() -> Self {
        Self {
            targets: BTreeSet::new(),
        }
    }

    pub fn perform_link(&mut self, insts: &[Inst], base_pc: u64) {
        let mut n = base_pc;

        let range = (base_pc..base_pc + (insts.len() as u64) * 4);

        let mut pc = base_pc;
        for inst in insts {
            pc += 4;

            match inst {
                Inst::Udf { .. } | Inst::Svc { .. } => break,
                Inst::Br { .. } | Inst::Blr { .. } | Inst::Ret { .. } => {
                    self.targets.insert(pc + 4);
                }
                Inst::B { label } | Inst::Bl { label } => {
                    self.targets.insert(pc + 4);
                    match pc.checked_add_signed(*label) {
                        Some(target) if range.contains(&target) => {
                            self.targets.insert(target);
                        }
                        _ => {}
                    };
                }
                _ => {}
            }
        }
    }

    #[inline]
    pub fn finalize(self) -> BTreeSet<u64> {
        self.targets
    }
}

#[cfg(test)]
mod tests {
    use crate::block_br::BranchAnalyzer;
    use crate::inst::Inst;

    #[test]
    fn test_link() {
        let mut b = BranchAnalyzer::new();

        let insts = [
            Inst::B { label: 4 },
            Inst::Nop,
            Inst::Nop,
            Inst::B { label: 8 },
            Inst::Nop,
            Inst::Nop,
            Inst::Nop,
            Inst::Nop,
        ];

        b.perform_link(&insts, 10000);

        println!("{:?}", b);
    }
}

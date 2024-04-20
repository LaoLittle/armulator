use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv8::a64::{InstDecoder, Instruction};

mod block_br;
pub mod context;
pub mod inst;
//mod mem;
mod error;
mod optimizer;
mod prepass;
mod translate;
mod translate_id;
mod aarch64;

struct AddressInstDecoder {
    addr: *const u32,
}

impl AddressInstDecoder {
    pub fn new(addr: usize) -> Self {
        Self { addr: addr as _ }
    }
}

impl Iterator for AddressInstDecoder {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let word = unsafe { *self.addr };

        unsafe {
            self.addr = self.addr.offset(1);
        }

        let bytes = word.to_le_bytes();
        let mut reader = U8Reader::new(&bytes);

        let inst = InstDecoder::default().decode(&mut reader).ok()?;

        Some(inst)
    }
}
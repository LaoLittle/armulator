use std::fmt::{Debug, Formatter, Write};

use onoff_core::error::{Error, Result};

pub trait Memory {
    type Addr;

    fn read8(&self, addr: Self::Addr) -> Result<u8>;

    fn write8(&mut self, addr: Self::Addr, src: u8) -> Result<()>;

    fn read16(&self, addr: Self::Addr) -> Result<u16>;

    fn write16(&mut self, addr: Self::Addr, src: u16) -> Result<()>;

    fn read32(&self, addr: Self::Addr) -> Result<u32>;

    fn write32(&mut self, addr: Self::Addr, src: u32) -> Result<()>;

    fn read64(&self, addr: Self::Addr) -> Result<u64>;

    fn write64(&mut self, addr: Self::Addr, src: u64) -> Result<()>;
}

pub const PAGE_SIZE: usize = 4096;

#[repr(transparent)]
pub struct Page(pub [u8; PAGE_SIZE]);

impl Page {
    #[inline]
    pub const fn new() -> Self {
        Self([0u8; PAGE_SIZE])
    }

    pub fn read_exact(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        let offset = offset as usize;
        let Some(mem) = self.0.get(offset..offset + buf.len()) else {
            return Err(Error::InvalidMemoryAddress);
        };

        buf.copy_from_slice(mem);

        Ok(())
    }

    pub fn write_exact(&mut self, buf: &[u8], offset: u64) -> Result<()> {
        let offset = offset as usize;
        let Some(mem) = self.0.get_mut(offset..offset + buf.len()) else {
            return Err(Error::InvalidMemoryAddress);
        };

        mem.copy_from_slice(buf);

        Ok(())
    }
}

impl Debug for Page {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("<Page>")
    }
}

const PAGE_SEGMENT_SIZE: usize = 512;

#[repr(transparent)]
struct PageLevel0(pub [Option<Box<Page>>; PAGE_SEGMENT_SIZE]);

impl PageLevel0 {
    #[inline]
    pub fn new() -> Self {
        Self(core::array::from_fn(|_| None))
    }
}

impl Debug for PageLevel0 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, p) in self.0.iter().enumerate() {
            if let Some(p) = p {
                f.write_char('[')?;
                write!(f, "<Page:{i}>")?;
                Debug::fmt(p, f)?;
                f.write_char(']')?;
            }
        }

        Ok(())
    }
}

struct PageLevel1(pub [Option<Box<PageLevel0>>; PAGE_SEGMENT_SIZE]);

impl PageLevel1 {
    #[inline]
    pub fn new() -> Self {
        Self(core::array::from_fn(|_| None))
    }
}

impl Debug for PageLevel1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, p0) in self.0.iter().enumerate() {
            if let Some(p0) = p0 {
                f.write_char('[')?;
                write!(f, "<Page0:{i}>")?;
                Debug::fmt(p0, f)?;
                f.write_char(']')?;
            }
        }

        Ok(())
    }
}

#[repr(transparent)]
struct PageLevel2(pub [Option<Box<PageLevel1>>; PAGE_SEGMENT_SIZE]);

impl PageLevel2 {
    #[inline]
    pub fn new() -> Self {
        Self(core::array::from_fn(|_| None))
    }
}

impl Debug for PageLevel2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, p1) in self.0.iter().enumerate() {
            if let Some(p1) = p1 {
                f.write_char('[')?;
                write!(f, "<Page1:{i}>")?;
                Debug::fmt(p1, f)?;
                f.write_char(']')?;
            }
        }

        Ok(())
    }
}

#[repr(transparent)]
struct PageLevel3(pub [Option<Box<PageLevel2>>; PAGE_SEGMENT_SIZE]);

impl PageLevel3 {
    pub fn new() -> Self {
        Self(core::array::from_fn(|_| None))
    }
}

impl Debug for PageLevel3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, p2) in self.0.iter().enumerate() {
            if let Some(p2) = p2 {
                f.write_char('[')?;
                write!(f, "<Page2:{i}>")?;
                Debug::fmt(p2, f)?;
                f.write_char(']')?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct PageMemory {
    pages: PageLevel3,
}

impl PageMemory {
    #[inline]
    pub fn new() -> Self {
        Self {
            pages: PageLevel3(core::array::from_fn(|_| None)),
        }
    }

    pub fn alloc_page(&mut self, addr: u64) -> &mut Page {
        let (p3, p2, p1, p0, _) = extract_addr(addr);

        let page = self.pages.0[p3].get_or_insert(PageLevel2::new().into()).0[p2]
            .get_or_insert(PageLevel1::new().into())
            .0[p1]
            .get_or_insert(PageLevel0::new().into())
            .0[p0]
            .get_or_insert(Page::new().into());

        page
    }

    // write bytes into certain page(s) provided by the addr argument
    // alloc if there were no available pages
    pub fn write_exact(&mut self, buf: &[u8], addr: u64) -> Result<()> {
        let (mut hi, lo) = align_4k(addr);

        let remain = PAGE_SIZE - lo as usize;

        // |Page .. ++| |Page ++ ..|
        for chunk in buf.chunks(PAGE_SIZE) {
            let (f, b) = if remain > chunk.len() {
                (chunk, [].as_slice())
            } else {
                chunk.split_at(remain)
            };

            self.alloc_page(hi).write_exact(f, lo)?;
            hi += 4096;

            if !b.is_empty() {
                self.alloc_page(hi).write_exact(b, 0)?;
            }
        }

        Ok(())
    }

    #[inline]
    pub fn get_page(&self, addr: u64) -> Result<&Page> {
        fn _get(pm: &PageLevel3, addr: u64) -> Option<&Page> {
            let (p3, p2, p1, p0, _) = extract_addr(addr);

            Some(
                pm.0.get(p3)?
                    .as_ref()?
                    .0
                    .get(p2)?
                    .as_ref()?
                    .0
                    .get(p1)?
                    .as_ref()?
                    .0
                    .get(p0)?
                    .as_ref()?
                    .as_ref(),
            )
        }

        //self.pages.get(&addr).ok_or(Error::InvalidMemoryAddress)
        _get(&self.pages, addr).ok_or(Error::InvalidMemoryAddress)
    }

    #[inline]
    pub fn get_page_mut(&mut self, addr: u64) -> Result<&mut Page> {
        fn _get(pm: &mut PageLevel3, addr: u64) -> Option<&mut Page> {
            let (p3, p2, p1, p0, _) = extract_addr(addr);

            Some(
                pm.0.get_mut(p3)?
                    .as_mut()?
                    .0
                    .get_mut(p2)?
                    .as_mut()?
                    .0
                    .get_mut(p1)?
                    .as_mut()?
                    .0
                    .get_mut(p0)?
                    .as_mut()?
                    .as_mut(),
            )
        }

        //self.pages.get_mut(&addr).ok_or(Error::InvalidMemoryAddress)
        _get(&mut self.pages, addr).ok_or(Error::InvalidMemoryAddress)
    }
}

impl Memory for PageMemory {
    type Addr = u64;

    fn read8(&self, addr: Self::Addr) -> Result<u8> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 1];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u8::from_le_bytes(buf))
    }

    fn write8(&mut self, addr: Self::Addr, src: u8) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read16(&self, addr: Self::Addr) -> Result<u16> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 2];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u16::from_le_bytes(buf))
    }

    fn write16(&mut self, addr: Self::Addr, src: u16) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read32(&self, addr: Self::Addr) -> Result<u32> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 4];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u32::from_le_bytes(buf))
    }

    fn write32(&mut self, addr: Self::Addr, src: u32) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read64(&self, addr: Self::Addr) -> Result<u64> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 8];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u64::from_le_bytes(buf))
    }

    fn write64(&mut self, addr: Self::Addr, src: u64) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }
}

// P[3], P[2], P[1], P[0], Page offset
const fn extract_addr(addr: u64) -> (usize, usize, usize, usize, usize) {
    const PAGE_OFFSET_MASK: u64 = 4096 - 1;
    const PAGE_SEGMENT_MASK: u64 = 512 - 1;

    (
        ((addr >> 39) & PAGE_SEGMENT_MASK) as usize,
        ((addr >> 30) & PAGE_SEGMENT_MASK) as usize,
        ((addr >> 21) & PAGE_SEGMENT_MASK) as usize,
        ((addr >> 12) & PAGE_SEGMENT_MASK) as usize,
        (addr & PAGE_OFFSET_MASK) as usize,
    )
}

// returns (hi, lo)
#[inline]
const fn align_4k(addr: u64) -> (u64, u64) {
    (addr & !4095, addr & 4095)
}

#[cfg(test)]
mod tests {
    use crate::mem::{Memory, PageMemory};
    use onoff_core::error::Result;

    #[test]
    fn page_mem() {
        let mut page_mem = PageMemory::new();

        for addr in [0, 4099, 10023] {
            page_mem.alloc_page(addr);

            mem_validation(&mut page_mem, addr).unwrap();
        }
    }

    fn mem_validation<M: Memory>(mem: &mut M, addr: M::Addr) -> Result<()>
    where
        M::Addr: Copy,
    {
        let a = 114;
        mem.write8(addr, a)?;
        let b = mem.read8(addr)?;
        assert_eq!(a, b);

        let a = 514;
        mem.write16(addr, a)?;
        let b = mem.read16(addr)?;
        assert_eq!(a, b);

        let a = 1919810;
        mem.write32(addr, a)?;
        let b = mem.read32(addr)?;
        assert_eq!(a, b);

        let a = 1145141919810;
        mem.write64(addr, a)?;
        let b = mem.read64(addr)?;
        assert_eq!(a, b);

        Ok(())
    }

    #[test]
    fn more_than_one_page() {
        let buf = [1u8; 8193];

        let mut pm = PageMemory::new();
        pm.write_exact(&buf, 0).unwrap();

        assert!(matches!(pm.read8(0), Ok(1)));
        assert!(matches!(pm.read8(4096), Ok(1)));
        assert!(matches!(pm.read8(8192), Ok(1)));
    }
}

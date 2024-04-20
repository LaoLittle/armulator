#[derive(Debug)]
pub enum Error {
    AlreadyLoaded,

    InvalidNsoHeader,
    InvalidNcaHeader,

    NotSupported,

    InvalidMemoryAddress,

    IO(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::IO(value)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

use std::{error::Error as StdError, fmt, io, path::PathBuf};

use serde_json::Error as SerdeError;

#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum Error {
    AccessError { path: PathBuf, inner: io::Error },
    CopyError { from: PathBuf, to: PathBuf, inner: io::Error },
    SerdeError { path: PathBuf, inner: SerdeError },
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AccessError { path, inner } => {
                write!(f, "Failed to access file {:?}: {}", path, inner)
            }
            Error::CopyError { from, to, inner } => {
                write!(f, "Failed to copy file {:?} to {:?}: {}", from, to, inner)
            }
            Error::SerdeError { path, inner } => write!(
                f,
                "Failed to read or write file {:?} due to serialization error: {}",
                path, inner
            ),
        }
    }
}
impl StdError for Error {
    fn description(&self) -> &str {
        match self {
            Error::AccessError { .. } => "AccessError",
            Error::CopyError { .. } => "CopyError",
            Error::SerdeError { .. } => "SerdeError",
        }
    }

    fn cause(&self) -> Option<&dyn StdError> {
        match self {
            Error::AccessError { inner, .. } => Some(inner),
            Error::CopyError { inner, .. } => Some(inner),
            Error::SerdeError { inner, .. } => Some(inner),
        }
    }
}

pub type Result<T> = ::std::result::Result<T, Error>;

pub(crate) fn log_error(e: &Error) {
    error!("error: {}", e);
}

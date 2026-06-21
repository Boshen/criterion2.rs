use std::{
    ffi::OsStr,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};

use crate::{
    error::{Error, Result},
    report::BenchmarkId,
};

pub fn load<A, P>(path: &P) -> Result<A>
where
    A: DeserializeOwned,
    P: AsRef<Path> + ?Sized,
{
    let path = path.as_ref();
    let mut f =
        File::open(path).map_err(|inner| Error::AccessError { inner, path: path.to_owned() })?;
    let mut string = String::new();
    let _ = f.read_to_string(&mut string);
    let result: A = serde_json::from_str(string.as_str())
        .map_err(|inner| Error::SerdeError { inner, path: path.to_owned() })?;

    Ok(result)
}

pub fn is_dir<P>(path: &P) -> bool
where
    P: AsRef<Path>,
{
    let path: &Path = path.as_ref();
    path.is_dir()
}

pub fn mkdirp<P>(path: &P) -> Result<()>
where
    P: AsRef<Path>,
{
    fs::create_dir_all(path.as_ref())
        .map_err(|inner| Error::AccessError { inner, path: path.as_ref().to_owned() })?;
    Ok(())
}

pub fn cp(from: &Path, to: &Path) -> Result<()> {
    fs::copy(from, to).map_err(|inner| Error::CopyError {
        inner,
        from: from.to_owned(),
        to: to.to_owned(),
    })?;
    Ok(())
}

pub fn save<D, P>(data: &D, path: &P) -> Result<()>
where
    D: Serialize,
    P: AsRef<Path>,
{
    let buf = serde_json::to_string(&data)
        .map_err(|inner| Error::SerdeError { path: path.as_ref().to_owned(), inner })?;
    save_string(&buf, path)
}

pub fn save_string<P>(data: &str, path: &P) -> Result<()>
where
    P: AsRef<Path>,
{
    use std::io::Write;

    File::create(path)
        .and_then(|mut f| f.write_all(data.as_bytes()))
        .map_err(|inner| Error::AccessError { inner, path: path.as_ref().to_owned() })?;

    Ok(())
}

pub fn list_existing_benchmarks<P>(directory: &P) -> Result<Vec<BenchmarkId>>
where
    P: AsRef<Path>,
{
    fn is_benchmark(path: &Path) -> bool {
        // Look for benchmark.json files inside folders named "new" (because we want to ignore
        // the baselines)
        path.file_name() == Some(OsStr::new("benchmark.json"))
            && path.parent().and_then(Path::file_name) == Some(OsStr::new("new"))
    }

    // Recursively collect every file under `directory`, ignoring any I/O errors
    // (e.g. unreadable directories) the same way the previous `walkdir` based
    // implementation did.
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(file_type) if file_type.is_dir() => visit(&path, files),
                Ok(_) => files.push(path),
                Err(_) => {}
            }
        }
    }

    let mut paths = vec![];
    visit(directory.as_ref(), &mut paths);

    let mut ids = vec![];
    for path in paths.into_iter().filter(|p| is_benchmark(p)) {
        let id: BenchmarkId = load(&path)?;
        ids.push(id);
    }

    Ok(ids)
}

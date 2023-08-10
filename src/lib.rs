pub mod splitter;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tiktoken_rs::{cl100k_base, p50k_base, p50k_edit, r50k_base};

#[pyclass]
struct RecursiveSplitterPy {
    inner: splitter::RecursiveSplitter,
}

#[pymethods]
impl RecursiveSplitterPy {
    #[new]
    #[pyo3(signature = (chunk_size, chunk_overlap = 0, tokenizer_kind = "len", tokenizer_subkind = "", separators = vec ! (), trim = false,))]
    fn new(chunk_size: usize, chunk_overlap: usize,
           tokenizer_kind: &str, tokenizer_subkind: &str,
           separators: Vec<String>, trim: bool) -> PyResult<Self> {
        let length_fn = match tokenizer_kind {
            "len" => splitter::LengthFunction::StrLen,
            "tiktoken" | "openai" => splitter::LengthFunction::TikToken({
                match tokenizer_subkind {
                    "" | "cl100k_base" => cl100k_base().map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                    "p50k_base" => p50k_base().map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                    "p50k_edit" => p50k_edit().map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                    "r50k_base" => r50k_base().map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                    s => return Err(PyValueError::new_err(format!("Unknown tokenizer subkind: {}", s))),
                }
            }),
            "huggingface" => splitter::LengthFunction::HuggingFace({
                // if tokenizer_subkind.starts_with('/') || tokenizer_subkind.starts_with('.') || tokenizer_subkind.contains(':') {
                tokenizers::Tokenizer::from_file(tokenizer_subkind).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                // } else {
                //     tokenizers::Tokenizer::from_pretrained(tokenizer_subkind, None).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                // }
            }),
            s => return Err(PyValueError::new_err(format!("Unknown tokenizer kind: {}", s))),
        };
        Ok(Self {
            inner: splitter::RecursiveSplitter::new(chunk_size, chunk_overlap, length_fn, separators, trim),
        })
    }

    pub fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        py.allow_threads(|| self.inner.count_tokens(text))
    }
    pub fn split_text(&self, py: Python<'_>, text: &str) -> Vec<String> {
        py.allow_threads(|| self.inner.split_text(text))
    }
}

#[pymodule]
fn recursive_split_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RecursiveSplitterPy>()?;
    Ok(())
}

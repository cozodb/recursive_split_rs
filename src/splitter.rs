use std::collections::VecDeque;
use tiktoken_rs::CoreBPE;
use tokenizers::tokenizer::Tokenizer;

#[derive(Default)]
pub enum LengthFunction {
    #[default]
    StrLen,
    TikToken(CoreBPE),
    HuggingFace(Tokenizer),
    Custom(Box<dyn Fn(&str) -> usize + Send + Sync>),
}

pub struct RecursiveSplitter {
    length_fn: Box<dyn Fn(&str) -> usize + Send + Sync>,
    separators: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    trim: bool,
}

impl RecursiveSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize, length_fn: LengthFunction, mut separators: Vec<String>, trim: bool) -> Self {
        let lf: Box<dyn Fn(&str) -> usize + Send + Sync> = match length_fn {
            LengthFunction::StrLen => Box::new(|s: &str| s.len()),
            LengthFunction::TikToken(bpe) => {
                let cls = move |s: &str| {
                    let tokens = bpe.encode_ordinary(s);
                    tokens.len()
                };
                Box::new(cls)
            }
            LengthFunction::HuggingFace(tokenizer) => {
                let cls = move |s: &str| {
                    let tokens = tokenizer.encode(s, false).unwrap_or_default();
                    tokens.len()
                };
                Box::new(cls)
            }
            LengthFunction::Custom(f) => f
        };
        if separators.is_empty() {
            separators = vec!["\n\n".to_string(),
                              "\n".to_string(),
                              "。".to_string(),
                              ". ".to_string(),
                              " ".to_string()]
        }
        separators.push("".to_string());
        Self {
            length_fn: lf,
            separators,
            chunk_size,
            chunk_overlap,
            trim,
        }
    }
    pub fn count_tokens(&self, text: &str) -> usize {
        (self.length_fn)(text)
    }
    pub fn split_text(&self, text: &str) -> Vec<String> {
        let ret = self.do_split_text(text, self.separators.clone());
        if self.trim {
            ret.into_iter().filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }).collect()
        } else {
            ret
        }
    }
    fn do_split_text(&self, text: &str, separators: Vec<String>) -> Vec<String> {
        let mut final_chunks: Vec<String> = vec![];
        let mut separator = separators.last().unwrap();
        let mut new_separators = vec![];
        // find the first separator that is in the text
        for (i, s) in separators.iter().enumerate() {
            if s.is_empty() {
                separator = s;
                break;
            }
            if text.trim().contains(s) {
                separator = s;
                new_separators = separators[i + 1..].to_vec();
                break;
            }
        }
        let splits = str_split_inclusive(text, separator);
        let mut good_splits = vec![];
        for s in splits {
            if (self.length_fn)(s) < self.chunk_size {
                good_splits.push(s)
            } else {
                if !good_splits.is_empty() {
                    self.merge_splits_and_collect(&mut final_chunks, &good_splits);
                    good_splits.clear();
                }

                if new_separators.is_empty() {
                    final_chunks.push(s.to_string());
                } else {
                    let new_chunks = self.do_split_text(s, new_separators.clone());
                    final_chunks.extend(new_chunks);
                }
            }
        }
        if !good_splits.is_empty() {
            self.merge_splits_and_collect(&mut final_chunks, &good_splits);
        }
        final_chunks
    }
    fn merge_splits_and_collect(&self, docs: &mut Vec<String>, splits: &[&str]) {
        let mut current_doc = VecDeque::new();
        let mut total = 0usize;
        for d in splits {
            let d_len = (self.length_fn)(d);
            if total + d_len > self.chunk_size {
                if self.chunk_overlap == 0 {
                    let doc = current_doc.into_iter().collect::<Vec<_>>().join("");
                    total = 0;
                    current_doc = VecDeque::new();
                    if !doc.is_empty() {
                        docs.push(doc);
                    }
                } else {
                    if !current_doc.is_empty() {
                        let doc = current_doc.iter().map(|s| *s).collect::<Vec<_>>().join("");
                        if !doc.is_empty() {
                            docs.push(doc);
                        }
                    }
                    while total > self.chunk_overlap || (
                        total + d_len > self.chunk_size && total > 0
                    ) {
                        let popped = current_doc.pop_front().unwrap();
                        total -= (self.length_fn)(popped);
                    }
                }
            }
            current_doc.push_back(d);
            total += d_len;
        }
        let doc = current_doc.into_iter().collect::<Vec<&str>>().join("");
        if !doc.is_empty() {
            docs.push(doc);
        }
    }
}


fn str_split_inclusive<'a>(text: &'a str, separator: &str) -> Vec<&'a str> {
    let chars: Vec<_> = separator.chars().collect();
    if chars.len() == 1 {
        text.split_inclusive(chars[0]).filter(|s| !s.is_empty()).collect()
    } else {
        text.split_inclusive(separator).filter(|s| !s.is_empty()).collect()
    }
}


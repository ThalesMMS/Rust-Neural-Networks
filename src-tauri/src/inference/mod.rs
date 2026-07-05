pub mod attention;
pub mod cifar10;
pub mod cnn;
pub mod mlp;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub probabilities: Vec<f32>,
    pub predicted_class: usize,
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn to_prediction(probabilities: Vec<f32>) -> Prediction {
    let predicted_class = argmax(&probabilities);
    Prediction { probabilities, predicted_class }
}

pub fn read_i32_le(bytes: &[u8], offset: &mut usize) -> Result<i32, String> {
    if *offset + 4 > bytes.len() {
        return Err("unexpected end of file while reading i32".to_string());
    }
    let v = i32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(v)
}

pub fn read_f32_le(bytes: &[u8], offset: &mut usize) -> Result<f32, String> {
    if *offset + 4 > bytes.len() {
        return Err("unexpected end of file while reading f32".to_string());
    }
    let v = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(v)
}

pub fn read_f32_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    (0..count).map(|_| read_f32_le(bytes, offset)).collect()
}

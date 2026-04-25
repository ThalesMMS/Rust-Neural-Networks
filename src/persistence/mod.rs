//! Shared checkpoint persistence helpers.

mod layer_io;

use crate::layers::Layer;
use crate::utils::rng::SimpleRng;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub use layer_io::{
    read_batchnorm_layer, read_conv2d_layer, read_dense_layer, read_dropout_layer,
    read_global_avgpool_layer, read_layer, read_residual_block, write_batchnorm_layer,
    write_conv2d_layer, write_dense_layer, write_dropout_layer, write_global_avgpool_layer,
    write_layer, write_residual_block,
};

/// Numeric layer IDs used by the layer-stack checkpoint format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerTypeId {
    Dense = 0,
    Conv2D = 1,
    BatchNorm = 2,
    Dropout = 3,
    ResidualBlock = 5,
    GlobalAvgPool = 6,
}

impl TryFrom<u8> for LayerTypeId {
    type Error = io::Error;

    fn try_from(value: u8) -> io::Result<Self> {
        Self::from_u8(value)
    }
}

impl From<LayerTypeId> for u8 {
    fn from(value: LayerTypeId) -> Self {
        value as u8
    }
}

impl LayerTypeId {
    /// Converts a raw checkpoint type byte into a layer type ID.
    pub fn from_u8(value: u8) -> io::Result<Self> {
        match value {
            0 => Ok(Self::Dense),
            1 => Ok(Self::Conv2D),
            2 => Ok(Self::BatchNorm),
            3 => Ok(Self::Dropout),
            5 => Ok(Self::ResidualBlock),
            6 => Ok(Self::GlobalAvgPool),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown layer type ID {value}"),
            )),
        }
    }
}

/// Writes a complete layer stack: u32 layer count followed by per-layer payloads.
pub fn save_layers<W: Write>(writer: &mut W, layers: &[Box<dyn Layer>]) -> io::Result<()> {
    writer.write_all(&(layers.len() as u32).to_le_bytes())?;
    for layer in layers {
        write_layer(writer, layer.as_ref())?;
    }
    Ok(())
}

/// Reads a complete layer stack written by [`save_layers`].
pub fn load_layers<R: Read>(
    reader: &mut R,
    rng: &mut SimpleRng,
) -> io::Result<Vec<Box<dyn Layer>>> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let num_layers = u32::from_le_bytes(buf4) as usize;

    let mut layers = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layers.push(read_layer(reader, rng)?);
    }
    Ok(layers)
}

/// File-based convenience wrapper around [`save_layers`].
pub fn save_layers_to_file<P: AsRef<Path>>(
    filename: P,
    layers: &[Box<dyn Layer>],
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    save_layers(&mut writer, layers)
}

/// File-based convenience wrapper around [`load_layers`].
pub fn load_layers_from_file<P: AsRef<Path>>(filename: P) -> io::Result<Vec<Box<dyn Layer>>> {
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut rng = SimpleRng::new(42);
    rng.reseed_from_time();
    load_layers(&mut reader, &mut rng)
}

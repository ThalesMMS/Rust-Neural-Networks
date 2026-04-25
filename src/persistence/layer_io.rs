use super::LayerTypeId;
use crate::layers::{
    BatchNormLayer, Conv2DLayer, DenseLayer, DropoutLayer, GlobalAvgPoolLayer, Layer, ResidualBlock,
};
use crate::utils::rng::SimpleRng;
use std::io::{self, Read, Write};

pub fn write_dense_layer<W: Write>(writer: &mut W, layer: &DenseLayer) -> io::Result<()> {
    writer.write_all(&(layer.input_size() as u32).to_le_bytes())?;
    writer.write_all(&(layer.output_size() as u32).to_le_bytes())?;
    write_f32_slice(writer, layer.weights())?;
    write_f32_slice(writer, layer.biases())
}

pub fn write_conv2d_layer<W: Write>(writer: &mut W, layer: &Conv2DLayer) -> io::Result<()> {
    writer.write_all(&(layer.in_channels() as u32).to_le_bytes())?;
    writer.write_all(&(layer.out_channels() as u32).to_le_bytes())?;
    writer.write_all(&(layer.kernel_size() as u32).to_le_bytes())?;
    writer.write_all(&(layer.padding() as i32).to_le_bytes())?;
    writer.write_all(&(layer.stride() as u32).to_le_bytes())?;
    writer.write_all(&(layer.input_height() as u32).to_le_bytes())?;
    writer.write_all(&(layer.input_width() as u32).to_le_bytes())?;
    write_f32_slice(writer, layer.weights())?;
    write_f32_slice(writer, layer.biases())
}

pub fn write_batchnorm_layer<W: Write>(writer: &mut W, layer: &BatchNormLayer) -> io::Result<()> {
    writer.write_all(&(layer.output_size() as u32).to_le_bytes())?;
    write_f32_slice(writer, layer.gamma())?;
    write_f32_slice(writer, layer.beta())?;
    write_f32_slice(writer, &layer.running_mean())?;
    write_f32_slice(writer, &layer.running_var())?;
    writer.write_all(&layer.epsilon().to_le_bytes())?;
    writer.write_all(&layer.momentum().to_le_bytes())
}

pub fn write_dropout_layer<W: Write>(writer: &mut W, layer: &DropoutLayer) -> io::Result<()> {
    writer.write_all(&(layer.output_size() as u32).to_le_bytes())?;
    writer.write_all(&layer.drop_rate().to_le_bytes())
}

/// Writes ResidualBlock architecture metadata only.
///
/// The block's internal Conv2D/BatchNorm weights are not exposed by the current
/// layer API, so this intentionally preserves the existing metadata-only format.
pub fn write_residual_block<W: Write>(writer: &mut W, block: &ResidualBlock) -> io::Result<()> {
    writer.write_all(&(block.in_channels() as u32).to_le_bytes())?;
    writer.write_all(&(block.out_channels() as u32).to_le_bytes())?;
    writer.write_all(&(block.out_height() as u32).to_le_bytes())?;
    writer.write_all(&(block.out_width() as u32).to_le_bytes())?;
    writer.write_all(&[block.has_projection_shortcut() as u8])
}

pub fn write_global_avgpool_layer<W: Write>(
    writer: &mut W,
    layer: &GlobalAvgPoolLayer,
) -> io::Result<()> {
    writer.write_all(&(layer.in_height() as u32).to_le_bytes())?;
    writer.write_all(&(layer.in_width() as u32).to_le_bytes())?;
    writer.write_all(&(layer.channels() as u32).to_le_bytes())
}

pub fn write_layer<W: Write>(writer: &mut W, layer: &dyn Layer) -> io::Result<()> {
    let any_layer = layer.as_any();

    if let Some(layer) = any_layer.downcast_ref::<DenseLayer>() {
        write_type_id(writer, LayerTypeId::Dense)?;
        write_dense_layer(writer, layer)
    } else if let Some(layer) = any_layer.downcast_ref::<Conv2DLayer>() {
        write_type_id(writer, LayerTypeId::Conv2D)?;
        write_conv2d_layer(writer, layer)
    } else if let Some(layer) = any_layer.downcast_ref::<BatchNormLayer>() {
        write_type_id(writer, LayerTypeId::BatchNorm)?;
        write_batchnorm_layer(writer, layer)
    } else if let Some(layer) = any_layer.downcast_ref::<DropoutLayer>() {
        write_type_id(writer, LayerTypeId::Dropout)?;
        write_dropout_layer(writer, layer)
    } else if let Some(layer) = any_layer.downcast_ref::<ResidualBlock>() {
        write_type_id(writer, LayerTypeId::ResidualBlock)?;
        write_residual_block(writer, layer)
    } else if let Some(layer) = any_layer.downcast_ref::<GlobalAvgPoolLayer>() {
        write_type_id(writer, LayerTypeId::GlobalAvgPool)?;
        write_global_avgpool_layer(writer, layer)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unknown layer type encountered during serialization",
        ))
    }
}

pub fn read_dense_layer<R: Read>(reader: &mut R) -> io::Result<Box<DenseLayer>> {
    let in_size = read_u32(reader)? as usize;
    let out_size = read_u32(reader)? as usize;
    let weights = read_f32_vec(reader, in_size * out_size)?;
    let biases = read_f32_vec(reader, out_size)?;
    Ok(Box::new(DenseLayer::new_with_weights(
        in_size, out_size, weights, biases,
    )))
}

pub fn read_conv2d_layer<R: Read>(reader: &mut R) -> io::Result<Box<Conv2DLayer>> {
    let in_channels = read_u32(reader)? as usize;
    let out_channels = read_u32(reader)? as usize;
    let kernel_size = read_u32(reader)? as usize;
    let padding = read_i32(reader)? as isize;
    let stride = read_u32(reader)? as usize;
    let input_height = read_u32(reader)? as usize;
    let input_width = read_u32(reader)? as usize;
    let weight_count = out_channels * in_channels * kernel_size * kernel_size;
    let weights = read_f32_vec(reader, weight_count)?;
    let biases = read_f32_vec(reader, out_channels)?;

    Ok(Box::new(Conv2DLayer::new_with_weights(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        input_height,
        input_width,
        weights,
        biases,
    )))
}

pub fn read_batchnorm_layer<R: Read>(reader: &mut R) -> io::Result<Box<BatchNormLayer>> {
    let size = read_u32(reader)? as usize;
    let gamma = read_f32_vec(reader, size)?;
    let beta = read_f32_vec(reader, size)?;
    let running_mean = read_f32_vec(reader, size)?;
    let running_var = read_f32_vec(reader, size)?;
    let epsilon = read_f32(reader)?;
    let momentum = read_f32(reader)?;

    Ok(Box::new(BatchNormLayer::new_with_params(
        size,
        epsilon,
        momentum,
        gamma,
        beta,
        running_mean,
        running_var,
    )))
}

pub fn read_dropout_layer<R: Read>(
    reader: &mut R,
    rng: &mut SimpleRng,
) -> io::Result<Box<DropoutLayer>> {
    let size = read_u32(reader)? as usize;
    let drop_rate = read_f32(reader)?;
    if !(0.0..1.0).contains(&drop_rate) {
        return Err(invalid_data(
            "dropout drop_rate must be in range [0.0, 1.0)",
        ));
    }
    Ok(Box::new(DropoutLayer::new(size, drop_rate, rng)))
}

pub fn read_residual_block<R: Read>(
    reader: &mut R,
    rng: &mut SimpleRng,
) -> io::Result<Box<ResidualBlock>> {
    let in_channels = read_u32(reader)? as usize;
    let out_channels = read_u32(reader)? as usize;
    let out_height = read_u32(reader)? as usize;
    let out_width = read_u32(reader)? as usize;
    let has_projection = read_u8(reader)? != 0;

    if in_channels == 0 || out_channels == 0 || out_height == 0 || out_width == 0 {
        return Err(invalid_data("residual block dimensions must be positive"));
    }

    let stride = if has_projection { 2 } else { 1 };
    let in_height = out_height * stride;
    let in_width = out_width * stride;

    Ok(Box::new(ResidualBlock::new(
        in_channels,
        out_channels,
        stride,
        in_height,
        in_width,
        rng,
    )))
}

pub fn read_global_avgpool_layer<R: Read>(reader: &mut R) -> io::Result<Box<GlobalAvgPoolLayer>> {
    let in_height = read_u32(reader)? as usize;
    let in_width = read_u32(reader)? as usize;
    let channels = read_u32(reader)? as usize;
    if in_height == 0 || in_width == 0 || channels == 0 {
        return Err(invalid_data(
            "global average pool dimensions must be positive",
        ));
    }
    Ok(Box::new(GlobalAvgPoolLayer::new(
        in_height, in_width, channels,
    )))
}

pub fn read_layer<R: Read>(reader: &mut R, rng: &mut SimpleRng) -> io::Result<Box<dyn Layer>> {
    match LayerTypeId::try_from(read_u8(reader)?)? {
        LayerTypeId::Dense => read_dense_layer(reader).map(|layer| layer as Box<dyn Layer>),
        LayerTypeId::Conv2D => read_conv2d_layer(reader).map(|layer| layer as Box<dyn Layer>),
        LayerTypeId::BatchNorm => read_batchnorm_layer(reader).map(|layer| layer as Box<dyn Layer>),
        LayerTypeId::Dropout => {
            read_dropout_layer(reader, rng).map(|layer| layer as Box<dyn Layer>)
        }
        LayerTypeId::ResidualBlock => {
            read_residual_block(reader, rng).map(|layer| layer as Box<dyn Layer>)
        }
        LayerTypeId::GlobalAvgPool => {
            read_global_avgpool_layer(reader).map(|layer| layer as Box<dyn Layer>)
        }
    }
}

fn write_type_id<W: Write>(writer: &mut W, layer_type: LayerTypeId) -> io::Result<()> {
    writer.write_all(&[u8::from(layer_type)])
}

fn write_f32_slice<W: Write>(writer: &mut W, values: &[f32]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f32_vec<R: Read>(reader: &mut R, len: usize) -> io::Result<Vec<f32>> {
    let mut values = vec![0.0; len];
    for value in &mut values {
        *value = read_f32(reader)?;
    }
    Ok(values)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

use knok::tensor::{Tensor1, Tensor2};

pub(crate) const SIZE: usize = 1024;
pub(crate) const PARTICLES: usize = 256;

pub(crate) type Field = Tensor2<f32, SIZE, SIZE>;
pub(crate) type ParticleVec = Tensor1<f32, PARTICLES>;
pub(crate) type AppResult<T> = std::result::Result<T, String>;

pub(crate) trait IntoAppResult<T> {
    fn into_app_result(self) -> AppResult<T>;
}

impl<T, E: ToString> IntoAppResult<T> for std::result::Result<T, E> {
    fn into_app_result(self) -> AppResult<T> {
        self.map_err(|error| error.to_string())
    }
}

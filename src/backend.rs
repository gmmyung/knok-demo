use knok::{Backend, Engine};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendChoice {
    Cpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(target_os = "macos")]
    Metal,
}

impl BackendChoice {
    pub(crate) fn available() -> Vec<Self> {
        let mut backends = vec![Self::Cpu];
        #[cfg(feature = "vulkan")]
        backends.push(Self::Vulkan);
        #[cfg(feature = "cuda")]
        backends.push(Self::Cuda);
        #[cfg(target_os = "macos")]
        backends.push(Self::Metal);
        backends
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "Vulkan",
            #[cfg(feature = "cuda")]
            Self::Cuda => "CUDA",
            #[cfg(target_os = "macos")]
            Self::Metal => "Metal",
        }
    }

    pub(crate) fn driver(self) -> &'static str {
        match self {
            Self::Cpu => "local-task",
            #[cfg(feature = "vulkan")]
            Self::Vulkan => "vulkan",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
            #[cfg(target_os = "macos")]
            Self::Metal => "metal",
        }
    }

    fn backend(self) -> Backend {
        match self {
            Self::Cpu => Backend::LlvmCpu,
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Backend::VulkanSpirv,
            #[cfg(feature = "cuda")]
            Self::Cuda => Backend::Cuda,
            #[cfg(target_os = "macos")]
            Self::Metal => Backend::MetalSpirv,
        }
    }
}

struct EngineSlot {
    engine: Option<Engine>,
    error: Option<String>,
}

impl EngineSlot {
    fn new() -> Self {
        Self {
            engine: None,
            error: None,
        }
    }

    fn get(&mut self, backend: Backend) -> std::result::Result<&Engine, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.engine.is_none() {
            match Engine::for_backend(backend) {
                Ok(engine) => self.engine = Some(engine),
                Err(error) => {
                    let message = error.to_string();
                    self.error = Some(message.clone());
                    return Err(message);
                }
            }
        }
        Ok(self.engine.as_ref().expect("engine was initialized"))
    }
}

pub(crate) struct EngineCache {
    cpu: EngineSlot,
    #[cfg(feature = "vulkan")]
    vulkan: EngineSlot,
    #[cfg(feature = "cuda")]
    cuda: EngineSlot,
    #[cfg(target_os = "macos")]
    metal: EngineSlot,
}

impl EngineCache {
    pub(crate) fn new() -> Self {
        Self {
            cpu: EngineSlot::new(),
            #[cfg(feature = "vulkan")]
            vulkan: EngineSlot::new(),
            #[cfg(feature = "cuda")]
            cuda: EngineSlot::new(),
            #[cfg(target_os = "macos")]
            metal: EngineSlot::new(),
        }
    }

    pub(crate) fn get(&mut self, backend: BackendChoice) -> std::result::Result<&Engine, String> {
        match backend {
            BackendChoice::Cpu => self.cpu.get(backend.backend()),
            #[cfg(feature = "vulkan")]
            BackendChoice::Vulkan => self.vulkan.get(backend.backend()),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => self.cuda.get(backend.backend()),
            #[cfg(target_os = "macos")]
            BackendChoice::Metal => self.metal.get(backend.backend()),
        }
    }
}

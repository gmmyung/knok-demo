#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tab {
    Mandelbrot,
    Heat,
    Wave,
    Life,
    Particles,
}

impl Tab {
    pub(crate) const ALL: [Self; 5] = [
        Self::Mandelbrot,
        Self::Heat,
        Self::Wave,
        Self::Life,
        Self::Particles,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Mandelbrot => "Mandelbrot",
            Self::Heat => "Heat",
            Self::Wave => "Wave",
            Self::Life => "Life",
            Self::Particles => "Particles",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MandelbrotIterations {
    Low,
    Medium,
    High,
}

impl MandelbrotIterations {
    pub(crate) const ALL: [Self; 3] = [Self::Low, Self::Medium, Self::High];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Low => "24",
            Self::Medium => "48",
            Self::High => "72",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiffusionPreset {
    Low,
    Medium,
    High,
}

impl DiffusionPreset {
    pub(crate) const ALL: [Self; 3] = [Self::Low, Self::Medium, Self::High];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WavePreset {
    Slow,
    Medium,
    Fast,
}

impl WavePreset {
    pub(crate) const ALL: [Self; 3] = [Self::Slow, Self::Medium, Self::Fast];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Slow => "Slow",
            Self::Medium => "Medium",
            Self::Fast => "Fast",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParticlePreset {
    Gentle,
    Strong,
}

impl ParticlePreset {
    pub(crate) const ALL: [Self; 2] = [Self::Gentle, Self::Strong];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Gentle => "Gentle",
            Self::Strong => "Strong",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ColorMap {
    Fire,
    Viridis,
    Ice,
}

impl ColorMap {
    pub(crate) const ALL: [Self; 3] = [Self::Fire, Self::Viridis, Self::Ice];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Fire => "Fire",
            Self::Viridis => "Viridis",
            Self::Ice => "Ice",
        }
    }
}

use crate::types::SIZE;

pub(crate) fn stamp_disc(
    field: &mut [f32],
    center_x: usize,
    center_y: usize,
    radius: usize,
    value: f32,
) {
    let radius2 = (radius * radius) as isize;
    let min_y = center_y.saturating_sub(radius);
    let max_y = (center_y + radius).min(SIZE - 1);
    let min_x = center_x.saturating_sub(radius);
    let max_x = (center_x + radius).min(SIZE - 1);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as isize - center_x as isize;
            let dy = y as isize - center_y as isize;
            if dx * dx + dy * dy <= radius2 {
                let index = y * SIZE + x;
                field[index] = (field[index] + value).clamp(-1.0, 1.0);
            }
        }
    }
}

pub(crate) fn stamp_life_cells(
    field: &mut [f32],
    center_x: usize,
    center_y: usize,
    radius: usize,
    value: f32,
) {
    let min_y = center_y.saturating_sub(radius);
    let max_y = (center_y + radius).min(SIZE - 1);
    let min_x = center_x.saturating_sub(radius);
    let max_x = (center_x + radius).min(SIZE - 1);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            field[y * SIZE + x] = value;
        }
    }
}

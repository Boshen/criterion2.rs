//! A collection of the most used traits, structs and enums

pub use super::candlestick::Candlesticks;
pub use super::curve::Curve::{Dots, Impulses, Lines, LinesPoints, Points, Steps};
pub use super::errorbar::ErrorBar::{XErrorBars, XErrorLines, YErrorBars, YErrorLines};
pub use super::filledcurve::FilledCurve;
pub use super::key::{Boxed, Horizontal, Justification, Order, Position, Stacked, Vertical};
pub use super::proxy::{Font, Label, Output, Title};
pub use super::traits::{Configure, Plot, Set};
pub use super::{
    Axes, Axis, BoxWidth, Color, Figure, FontSize, Grid, Key, LineType, LineWidth, Opacity,
    PointSize, PointType, Range, Scale, ScaleFactor, Size, Terminal, TicLabels,
};

//! Shared timeline for audio levels and buffer water level.
//!
//! Every processed frame appends one row. The rows are dumped to CSV for
//! analysis and rendered to a dependency-free SVG so the operator can see
//! audio level and buffer water level on one common time axis.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

use crate::delay::EngineState;
use crate::png::Canvas;

#[derive(Clone, Copy, Debug)]
pub struct TimelineRow {
    pub t: f32,
    pub water_level: f32,
    pub in_rms: f32,
    pub out_rms: f32,
    pub state: EngineState,
    pub dumped: bool,
    pub held: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Event {
    /// Operator pressed the dump key at this time.
    Dump { t: f32 },
    /// A speech-substitute marker occupies [start, end) in the input.
    Marker { start: f32, end: f32 },
}

#[derive(Default)]
pub struct Timeline {
    pub rows: Vec<TimelineRow>,
    pub events: Vec<Event>,
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-6).log10()
}

impl Timeline {
    pub fn push(&mut self, row: TimelineRow) {
        self.rows.push(row);
    }

    pub fn write_csv(&self, path: &Path) -> io::Result<()> {
        let mut s = String::from("t,water_level_s,in_rms_db,out_rms_db,state,dumped,held\n");
        for r in &self.rows {
            let state = match r.state {
                EngineState::Steady => "steady",
                EngineState::Dumping => "dumping",
                EngineState::Rebuilding => "rebuilding",
            };
            let _ = writeln!(
                s,
                "{:.3},{:.4},{:.1},{:.1},{},{},{}",
                r.t,
                r.water_level,
                db(r.in_rms),
                db(r.out_rms),
                state,
                r.dumped as u8,
                r.held as u8,
            );
        }
        fs::write(path, s)
    }

    pub fn write_svg(&self, path: &Path, target_delay: f32) -> io::Result<()> {
        fs::write(path, self.render_svg(target_delay))
    }

    /// Raster twin of the SVG: same shared time axis, same panels.
    /// Panel 1 (top): buffer water level, target dashed, rebuild shading.
    /// Panel 2: input (green) and speaker-out (red) levels in dBFS.
    /// Orange verticals: dump-key presses. Red shading: marker intervals.
    pub fn write_png(&self, path: &Path, target_delay: f32) -> io::Result<()> {
        const W: usize = 1200;
        const H: usize = 480;
        const LEFT: f32 = 60.0;
        const TOP1: f32 = 40.0;
        const PANEL_H: f32 = 180.0;
        const GAP: f32 = 30.0;
        let plot_w = W as f32 - LEFT - 20.0;
        let top2 = TOP1 + PANEL_H + GAP;
        let t_max = self.rows.last().map(|r| r.t).unwrap_or(1.0).max(1.0);
        let x = |t: f32| LEFT + plot_w * t / t_max;
        let y_wl = |v: f32| TOP1 + PANEL_H - PANEL_H * (v / (target_delay * 1.3)).min(1.0);
        let y_db = |v: f32| top2 + PANEL_H - PANEL_H * ((v + 80.0) / 80.0).clamp(0.0, 1.0);

        let white = (255, 255, 255);
        let mut c = Canvas::new(W, H, white);
        // Panel backgrounds.
        c.rect(LEFT as i32, TOP1 as i32, (LEFT + plot_w) as i32, (TOP1 + PANEL_H) as i32, (250, 250, 250));
        c.rect(LEFT as i32, top2 as i32, (LEFT + plot_w) as i32, (top2 + PANEL_H) as i32, (250, 250, 250));
        // Rebuild (held-frame) shading on the water panel.
        for r in self.rows.iter().filter(|r| r.held) {
            c.rect(x(r.t) as i32, TOP1 as i32, x(r.t) as i32 + 2, (TOP1 + PANEL_H) as i32, (223, 240, 216));
        }
        // Target delay dashed line.
        let yt = y_wl(target_delay) as i32;
        let mut dx = LEFT as i32;
        while dx < (LEFT + plot_w) as i32 {
            c.rect(dx, yt, dx + 6, yt + 1, (136, 136, 136));
            dx += 10;
        }
        // Marker intervals on the audio panel; dump presses on both panels.
        for ev in &self.events {
            match *ev {
                Event::Dump { t } => {
                    let ex = x(t) as i32;
                    c.rect(ex, TOP1 as i32, ex + 2, (TOP1 + PANEL_H) as i32, (255, 127, 14));
                    c.rect(ex, top2 as i32, ex + 2, (top2 + PANEL_H) as i32, (255, 127, 14));
                }
                Event::Marker { start, end } => {
                    c.rect_blend(
                        x(start) as i32,
                        top2 as i32,
                        (x(end) as i32).max(x(start) as i32 + 2),
                        (top2 + PANEL_H) as i32,
                        (214, 39, 40),
                        0.18,
                    );
                }
            }
        }
        // Water-level polyline.
        let wl: Vec<(f32, f32)> = self.rows.iter().map(|r| (x(r.t), y_wl(r.water_level))).collect();
        c.polyline(&wl, (31, 119, 180));
        // Audio level polylines.
        let inp: Vec<(f32, f32)> = self.rows.iter().map(|r| (x(r.t), y_db(db(r.in_rms)))).collect();
        let outp: Vec<(f32, f32)> = self.rows.iter().map(|r| (x(r.t), y_db(db(r.out_rms)))).collect();
        c.polyline(&inp, (44, 160, 44));
        c.polyline(&outp, (214, 39, 40));
        // Panel borders and time axis.
        let gray = (204, 204, 204);
        for ty in [TOP1, top2] {
            c.line(LEFT, ty, LEFT + plot_w, ty, gray);
            c.line(LEFT, ty + PANEL_H, LEFT + plot_w, ty + PANEL_H, gray);
            c.line(LEFT, ty, LEFT, ty + PANEL_H, gray);
            c.line(LEFT + plot_w, ty, LEFT + plot_w, ty + PANEL_H, gray);
        }
        let axis_y = top2 + PANEL_H + 20.0;
        c.line(LEFT, axis_y, LEFT + plot_w, axis_y, (51, 51, 51));
        let mut t = 0.0;
        while t <= t_max {
            c.line(x(t), axis_y, x(t), axis_y + 5.0, (51, 51, 51));
            t += (t_max / 12.0).max(1.0);
        }
        c.write(path)
    }

    fn render_svg(&self, target_delay: f32) -> String {
        let w = 1200.0f32;
        let panel_h = 180.0f32;
        let gap = 30.0f32;
        let top = 40.0f32;
        let left = 60.0f32;
        let plot_w = w - left - 20.0;
        let h = top + panel_h * 2.0 + gap + 60.0;
        let t_max = self.rows.last().map(|r| r.t).unwrap_or(1.0).max(1.0);
        let x = |t: f32| left + plot_w * t / t_max;

        let mut s = String::new();
        let _ = writeln!(
            s,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             font-family=\"monospace\" font-size=\"12\">"
        );
        let _ = writeln!(s, "<rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>");
        let _ = writeln!(
            s,
            "<text x=\"{left}\" y=\"24\" font-size=\"16\">profanity-delay: audio &amp; buffer water level — shared timeline</text>"
        );

        // Panel 1: water level.
        let y1 = top + panel_h;
        let y_wl = |v: f32| y1 - panel_h * (v / (target_delay * 1.3)).min(1.0);
        let _ = writeln!(
            s,
            "<rect x=\"{left}\" y=\"{top}\" width=\"{plot_w}\" height=\"{panel_h}\" fill=\"#fafafa\" stroke=\"#ccc\"/>"
        );
        // Held-frame shading (rebuild activity).
        for r in self.rows.iter().filter(|r| r.held) {
            let _ = writeln!(
                s,
                "<rect x=\"{:.1}\" y=\"{top}\" width=\"2\" height=\"{panel_h}\" fill=\"#dff0d8\"/>",
                x(r.t)
            );
        }
        let y_t = y_wl(target_delay);
        let _ = writeln!(
            s,
            "<line x1=\"{left}\" y1=\"{y_t:.1}\" x2=\"{:.1}\" y2=\"{y_t:.1}\" stroke=\"#888\" stroke-dasharray=\"6 4\"/>",
            left + plot_w
        );
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#666\">target {target_delay:.1}s</text>",
            left + plot_w - 90.0,
            y_t - 4.0
        );
        let pts: String = self
            .rows
            .iter()
            .map(|r| format!("{:.1},{:.1}", x(r.t), y_wl(r.water_level)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            s,
            "<polyline points=\"{pts}\" fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1.5\"/>"
        );
        let _ = writeln!(
            s,
            "<text x=\"8\" y=\"{:.1}\">water</text><text x=\"8\" y=\"{:.1}\">level(s)</text>",
            top + 60.0,
            top + 76.0
        );

        // Panel 2: audio levels (dB).
        let top2 = top + panel_h + gap;
        let y2 = top2 + panel_h;
        let y_db = |v: f32| y2 - panel_h * ((v + 80.0) / 80.0).clamp(0.0, 1.0);
        let _ = writeln!(
            s,
            "<rect x=\"{left}\" y=\"{top2}\" width=\"{plot_w}\" height=\"{panel_h}\" fill=\"#fafafa\" stroke=\"#ccc\"/>"
        );
        let in_pts: String = self
            .rows
            .iter()
            .map(|r| format!("{:.1},{:.1}", x(r.t), y_db(db(r.in_rms))))
            .collect::<Vec<_>>()
            .join(" ");
        let out_pts: String = self
            .rows
            .iter()
            .map(|r| format!("{:.1},{:.1}", x(r.t), y_db(db(r.out_rms))))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            s,
            "<polyline points=\"{in_pts}\" fill=\"none\" stroke=\"#2ca02c\" stroke-width=\"1\"/>"
        );
        let _ = writeln!(
            s,
            "<polyline points=\"{out_pts}\" fill=\"none\" stroke=\"#d62728\" stroke-width=\"1\"/>"
        );
        let _ = writeln!(
            s,
            "<text x=\"8\" y=\"{:.1}\">audio</text><text x=\"8\" y=\"{:.1}\">dBFS</text>",
            top2 + 60.0,
            top2 + 76.0
        );
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#2ca02c\">input</text>\
             <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#d62728\">speaker out</text>",
            left + plot_w - 160.0,
            top2 + 16.0,
            left + plot_w - 90.0,
            top2 + 16.0
        );

        // Events on both panels.
        for ev in &self.events {
            match *ev {
                Event::Dump { t } => {
                    let ex = x(t);
                    for ty in [top, top2] {
                        let _ = writeln!(
                            s,
                            "<line x1=\"{ex:.1}\" y1=\"{ty}\" x2=\"{ex:.1}\" y2=\"{:.1}\" stroke=\"#ff7f0e\" stroke-width=\"2\"/>",
                            ty + panel_h
                        );
                    }
                    let _ = writeln!(
                        s,
                        "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#ff7f0e\">DUMP</text>",
                        ex + 3.0,
                        top + 12.0
                    );
                }
                Event::Marker { start, end } => {
                    let _ = writeln!(
                        s,
                        "<rect x=\"{:.1}\" y=\"{top2}\" width=\"{:.1}\" height=\"{panel_h}\" fill=\"#d62728\" fill-opacity=\"0.15\"/>",
                        x(start),
                        (x(end) - x(start)).max(2.0)
                    );
                    let _ = writeln!(
                        s,
                        "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#d62728\">marker</text>",
                        x(start),
                        top2 + panel_h - 4.0
                    );
                }
            }
        }

        // Time axis.
        let axis_y = top2 + panel_h + 20.0;
        let _ = writeln!(
            s,
            "<line x1=\"{left}\" y1=\"{axis_y}\" x2=\"{:.1}\" y2=\"{axis_y}\" stroke=\"#333\"/>",
            left + plot_w
        );
        let mut t = 0.0;
        while t <= t_max {
            let _ = writeln!(
                s,
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{t:.0}s</text>",
                x(t),
                axis_y + 14.0
            );
            t += (t_max / 12.0).max(1.0);
        }
        s.push_str("</svg>\n");
        s
    }
}

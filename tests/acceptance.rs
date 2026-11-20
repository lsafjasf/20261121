use roundtrip_probe::{
    ChunkPattern, FalseBurst, FrameObservation, LoopbackConfig, MatcherConfig, MeasurementSession,
    Probe, ProbeConfig, SoftwareLoopback, TrackerConfig,
};

fn run_scenario(
    delays: Vec<f32>,
    chunks: Vec<usize>,
    dropout_frames: Vec<usize>,
    false_bursts: Vec<FalseBurst>,
) -> Vec<FrameObservation> {
    let probe = Probe::new(ProbeConfig {
        amplitude: 0.008,
        ..ProbeConfig::default()
    })
    .unwrap();
    let config = LoopbackConfig {
        frames: delays.len(),
        delays,
        echo_gain: 0.8,
        noise_rms: 0.010,
        interference_rms: 0.050,
        interference_seed: ProbeConfig::default().seed,
        chunks: ChunkPattern::new(chunks).unwrap(),
        dropout_frames,
        false_bursts,
    };
    let render = SoftwareLoopback::new(probe.clone(), config)
        .unwrap()
        .render();
    let matcher_config = MatcherConfig {
        max_lag: Some(probe.max_lag()),
        ..MatcherConfig::default()
    };
    let mut session = MeasurementSession::new(&probe, matcher_config, TrackerConfig::default());
    let mut observations = Vec::new();
    let mut offset = 0;
    for size in &render.chunks {
        let end = offset + size;
        observations.extend(session.push(&render.capture[offset..end]));
        offset = end;
    }
    observations
}

#[test]
fn known_delay_steps_track_through_noise_and_uneven_buffers() {
    let mut delays = Vec::new();
    for center in [85.0_f32, 160.0, 55.0] {
        for frame in 0..30 {
            delays.push(
                center + 0.65 * (frame as f32 * 1.37).sin() + 0.35 * (frame as f32 * 0.31).sin(),
            );
        }
    }
    let observations = run_scenario(
        delays.clone(),
        vec![257, 31, 1_024, 93, 511, 7, 333, 1, 191],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(observations.len(), delays.len());

    for start in [0, 30, 60] {
        let settled: Vec<_> = observations[start + 7..start + 30]
            .iter()
            .filter(|observation| observation.accepted)
            .collect();
        assert!(
            settled.len() >= 19,
            "segment {start} accepted only {} frames",
            settled.len()
        );
        for observation in settled {
            let error = (observation.reported_lag.unwrap() - delays[observation.frame_index]).abs();
            assert!(
                error < 2.0,
                "frame {} had error {error}",
                observation.frame_index
            );
        }
    }
}

#[test]
fn one_false_peak_with_muted_echo_does_not_poison_tracking() {
    let delays = vec![80.0; 30];
    let observations = run_scenario(
        delays,
        vec![257, 31, 1_024, 93, 511, 7, 333],
        vec![15],
        vec![FalseBurst {
            frame: 15,
            lag: 600,
            amplitude: 0.30,
        }],
    );
    assert!(!observations[15].accepted);
    assert!(observations[16].accepted);
    assert!((observations[16].reported_lag.unwrap() - 80.0).abs() < 1.5);
    assert!(
        observations
            .iter()
            .filter(|observation| observation.accepted)
            .all(|observation| (observation.reported_lag.unwrap() - 600.0).abs() > 10.0)
    );
}

#[test]
fn buffer_chunking_does_not_change_observations() {
    let delays = vec![120.25; 18];
    let regular = run_scenario(delays.clone(), vec![512], Vec::new(), Vec::new());
    let irregular = run_scenario(
        delays,
        vec![1, 37, 900, 5, 1_574, 41, 2_051, 88, 3_111],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(regular.len(), irregular.len());
    for (left, right) in regular.iter().zip(&irregular) {
        assert_eq!(left.window, right.window);
        assert_eq!(left.reported_lag, right.reported_lag);
        assert_eq!(left.state, right.state);
        assert!((left.confidence - right.confidence).abs() < 1.0e-6);
    }
}

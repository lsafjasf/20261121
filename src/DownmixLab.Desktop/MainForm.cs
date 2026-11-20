using DownmixLab.Core.Analysis;
using DownmixLab.Core.Audio;
using DownmixLab.Core.Downmix;
using DownmixLab.Core.Material;
using DownmixLab.Core.Session;
using DownmixLab.Core.Snapshot;
using DownmixLab.Core.Verification;

namespace DownmixLab.Desktop;

public sealed class MainForm : Form
{
    private readonly DataGridView _gridA = MakeMatrixGrid();
    private readonly DataGridView _gridB = MakeMatrixGrid();
    private readonly ComboBox _lfeMode = new() { DropDownStyle = ComboBoxStyle.DropDownList };
    private readonly NumericUpDown _lfeAttenuation = new() { Minimum = 0, Maximum = 60, Value = 10, DecimalPlaces = 1 };
    private readonly NumericUpDown _segmentCount = new() { Minimum = 1, Maximum = 64, Value = 4 };
    private readonly DataGridView _results = new() { Dock = DockStyle.Fill, ReadOnly = true, AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill };
    private readonly Label _status = new() { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };

    private AudioBuffer? _source;
    private string? _sourcePath;
    private MonitoringSession? _session;

    public MainForm()
    {
        Text = "DownmixLab - 5.1 downmix monitoring";
        Width = 1180;
        Height = 760;

        var layout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4 };
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        Controls.Add(layout);

        layout.Controls.Add(BuildToolbar(), 0, 0);
        layout.Controls.Add(BuildMatrixPanel(), 0, 1);
        layout.Controls.Add(_results, 0, 2);
        layout.Controls.Add(_status, 0, 3);

        LoadMatrix(_gridA, DownmixMatrix.ItuStereo());
        LoadMatrix(_gridB, DownmixMatrix.DialogueFriendly());
        _lfeMode.Items.AddRange(new object[] { "Exclude", "Include", "Attenuate" });
        _lfeMode.SelectedIndex = 0;
    }

    private Control BuildToolbar()
    {
        var bar = new FlowLayoutPanel { Dock = DockStyle.Top, AutoSize = true };
        bar.Controls.Add(MakeButton("Open WAV/BWF...", OpenFile));
        bar.Controls.Add(MakeButton("Built-in material", LoadBuiltin));
        bar.Controls.Add(MakeButton("Run A/B", RunAnalysis));
        bar.Controls.Add(MakeButton("Export monitor WAVs", ExportWavs));
        bar.Controls.Add(MakeButton("Save snapshot", SaveSnapshot));
        bar.Controls.Add(MakeButton("Verify vectors", VerifyVectors));
        bar.Controls.Add(new Label { Text = "Segments:", AutoSize = true, Margin = new Padding(12, 8, 2, 2) });
        bar.Controls.Add(_segmentCount);
        bar.Controls.Add(new Label { Text = "LFE:", AutoSize = true, Margin = new Padding(12, 8, 2, 2) });
        bar.Controls.Add(_lfeMode);
        bar.Controls.Add(new Label { Text = "dB att:", AutoSize = true, Margin = new Padding(6, 8, 2, 2) });
        bar.Controls.Add(_lfeAttenuation);
        return bar;
    }

    private Control BuildMatrixPanel()
    {
        var panel = new TableLayoutPanel { Dock = DockStyle.Top, AutoSize = true, ColumnCount = 2 };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        panel.Controls.Add(WrapGrid("Matrix A", _gridA), 0, 0);
        panel.Controls.Add(WrapGrid("Matrix B", _gridB), 1, 0);
        return panel;
    }

    private static Control WrapGrid(string title, DataGridView grid)
    {
        var box = new GroupBox { Text = title, Dock = DockStyle.Top, Height = 220 };
        grid.Dock = DockStyle.Fill;
        box.Controls.Add(grid);
        return box;
    }

    private static DataGridView MakeMatrixGrid()
    {
        var grid = new DataGridView
        {
            AllowUserToAddRows = false,
            RowHeadersVisible = false,
            ColumnCount = 2,
        };
        grid.Columns[0].Name = "Lt";
        grid.Columns[1].Name = "Rt";
        foreach (var unused in DownmixMatrix.InputNames)
            grid.Rows.Add(0.0, 0.0);
        for (int i = 0; i < DownmixMatrix.InputNames.Length; i++)
            grid.Rows[i].HeaderCell.Value = DownmixMatrix.InputNames[i];
        grid.RowHeadersVisible = true;
        return grid;
    }

    private static Button MakeButton(string text, EventHandler onClick)
    {
        var button = new Button { Text = text, AutoSize = true };
        button.Click += onClick;
        return button;
    }

    private static void LoadMatrix(DataGridView grid, DownmixMatrix matrix)
    {
        for (int i = 0; i < DownmixMatrix.InputChannels; i++)
            for (int o = 0; o < DownmixMatrix.OutputChannels; o++)
                grid.Rows[i].Cells[o].Value = matrix[i, o];
    }

    private static DownmixMatrix ReadMatrix(DataGridView grid)
    {
        var m = new DownmixMatrix();
        for (int i = 0; i < DownmixMatrix.InputChannels; i++)
            for (int o = 0; o < DownmixMatrix.OutputChannels; o++)
                m[i, o] = Convert.ToDouble(grid.Rows[i].Cells[o].Value ?? 0.0);
        return m;
    }

    private LfePolicy CurrentLfePolicy() => new()
    {
        Mode = (LfeInclusionMode)_lfeMode.SelectedIndex,
        AttenuationDb = (double)_lfeAttenuation.Value,
    };

    private void OpenFile(object? sender, EventArgs e)
    {
        using var dialog = new OpenFileDialog { Filter = "WAV/BWF files|*.wav;*.bwf|All files|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK) return;
        try
        {
            var wav = WavFile.Read(dialog.FileName);
            if (wav.Buffer.Channels != 6)
            {
                MessageBox.Show(this, "Input must be 5.1 (6 channels: L R C LFE Ls Rs).",
                    "Unsupported layout", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            _source = wav.Buffer;
            _sourcePath = dialog.FileName;
            _status.Text = $"Loaded {Path.GetFileName(dialog.FileName)}: 6ch {wav.Buffer.SampleRate} Hz, " +
                $"{wav.Buffer.Frames} frames, {wav.Format}{(wav.IsBwf ? ", BWF" : "")}";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Failed to read file",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void LoadBuiltin(object? sender, EventArgs e)
    {
        _source = TestMaterialGenerator.Generate();
        _sourcePath = null;
        _status.Text = "Built-in material loaded: center dialogue / inverted surrounds / full mix.";
    }

    private void RunAnalysis(object? sender, EventArgs e)
    {
        if (_source == null)
        {
            _status.Text = "Load a WAV/BWF file or the built-in material first.";
            return;
        }
        try
        {
            _session = new MonitoringSession(_source, ReadMatrix(_gridA), ReadMatrix(_gridB),
                CurrentLfePolicy());
            var segments = _sourcePath == null
                ? TestMaterialGenerator.DefaultSegments(_source.SampleRate)
                : MonitoringSession.EqualSegments(_source, (int)_segmentCount.Value);
            _session.Run(segments);
            ShowResults();
            _status.Text = $"Analysis done. LFE policy: {_session.LfePolicy}";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Analysis failed",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    private void ShowResults()
    {
        _results.Columns.Clear();
        _results.Rows.Clear();
        string[] cols = { "Segment", "Metric", "A", "B", "A-B" };
        foreach (var c in cols) _results.Columns.Add(c, c);
        if (_session == null) return;
        foreach (var cmp in _session.Comparisons)
        {
            void Row(string metric, double a, double b) =>
                _results.Rows.Add(cmp.Segment.Label, metric,
                    a.ToString("F3"), b.ToString("F3"), (a - b).ToString("F3"));
            Row("correlation", cmp.A.Correlation, cmp.B.Correlation);
            Row("mid/side dB", cmp.A.MidSideRatioDb, cmp.B.MidSideRatioDb);
            Row("pan", cmp.A.PanPosition, cmp.B.PanPosition);
            Row("mono sum loss dB", cmp.A.MonoSumLossDb, cmp.B.MonoSumLossDb);
            Row("cancelled ratio", cmp.A.CancelledSampleRatio, cmp.B.CancelledSampleRatio);
            Row("center->Lt dB", cmp.A.CenterGainLtDb, cmp.B.CenterGainLtDb);
            Row("center->Rt dB", cmp.A.CenterGainRtDb, cmp.B.CenterGainRtDb);
            foreach (var kv in cmp.A.BandEnergyDb)
                Row($"band {kv.Key} Hz dB", kv.Value, cmp.B.BandEnergyDb[kv.Key]);
        }
    }

    private void ExportWavs(object? sender, EventArgs e)
    {
        if (_session?.MonitorA == null)
        {
            _status.Text = "Run the analysis first.";
            return;
        }
        using var dialog = new FolderBrowserDialog();
        if (dialog.ShowDialog(this) != DialogResult.OK) return;
        string a = Path.Combine(dialog.SelectedPath, "monitor_A.wav");
        string b = Path.Combine(dialog.SelectedPath, "monitor_B.wav");
        WavWriter.Write(a, _session.MonitorA, WavSampleFormat.Pcm24);
        WavWriter.Write(b, _session.MonitorB!, WavSampleFormat.Pcm24);
        _status.Text = $"Exported {a} and {b}";
    }

    private void SaveSnapshot(object? sender, EventArgs e)
    {
        if (_session == null)
        {
            _status.Text = "Run the analysis first.";
            return;
        }
        using var dialog = new SaveFileDialog { Filter = "JSON snapshot|*.json", FileName = "matrix_snapshot.json" };
        if (dialog.ShowDialog(this) != DialogResult.OK) return;
        MatrixSnapshot.Capture(_session, _sourcePath).Save(dialog.FileName);
        _status.Text = $"Snapshot saved to {dialog.FileName}";
    }

    private void VerifyVectors(object? sender, EventArgs e)
    {
        var results = KnownVectorVerifier.RunAll();
        var lines = results.Select(r =>
            $"{(r.Passed ? "PASS" : "FAIL")}  {r.Name}  (max err {r.MaxAbsError:E2})");
        MessageBox.Show(this, string.Join(Environment.NewLine, lines),
            results.All(r => r.Passed) ? "All vectors verified" : "Verification failed",
            MessageBoxButtons.OK,
            results.All(r => r.Passed) ? MessageBoxIcon.Information : MessageBoxIcon.Error);
    }
}

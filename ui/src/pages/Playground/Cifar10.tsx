import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud } from "lucide-react";
import { api, type ModelDescriptor, type Prediction } from "../../lib/tauri";
import ProbabilityBars from "../../components/ProbabilityBars";

const CLASS_LABELS = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
];

export default function Cifar10() {
  const previewRef = useRef<HTMLCanvasElement>(null);
  const navigate = useNavigate();
  const [model, setModel] = useState<ModelDescriptor | null>(null);
  const [trainedYet, setTrainedYet] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  useEffect(() => {
    api.listModels().then(async (all) => {
      const m = all.find((x) => x.id === "cifar10_cnn") ?? null;
      setModel(m);
      if (m?.checkpoints[0]) {
        const [ok] = await api.checkpointStatus([m.checkpoints[0]]);
        setTrainedYet(ok);
      }
    });
  }, []);

  async function handleFile(file: File) {
    setError(null);
    setFileName(file.name);
    const bitmap = await createImageBitmap(file);
    const canvas = previewRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, 32, 32);
    ctx.drawImage(bitmap, 0, 0, 32, 32);
    const data = ctx.getImageData(0, 0, 32, 32).data;

    const pixels: number[] = new Array(32 * 32 * 3);
    for (let i = 0; i < 32 * 32; i++) {
      pixels[i * 3] = data[i * 4] / 255;
      pixels[i * 3 + 1] = data[i * 4 + 1] / 255;
      pixels[i * 3 + 2] = data[i * 4 + 2] / 255;
    }

    if (!model?.checkpoints[0] || !trainedYet) return;
    try {
      const result = await api.predictCifar10(model.checkpoints[0], pixels);
      setPrediction(result);
    } catch (e) {
      setError(String(e));
    }
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  return (
    <div className="mx-auto max-w-4xl px-8 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-zinc-100">CIFAR-10 image classification</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Upload a photo — it's resized to 32×32 and classified by the CIFAR-10 CNN.
        </p>
      </header>

      {!trainedYet ? (
        <div className="card flex flex-col items-start gap-2 p-6 text-sm text-zinc-400">
          The CIFAR-10 CNN hasn't been trained yet.
          <button
            onClick={() => navigate("/train/cifar10_cnn")}
            className="rounded-md bg-accent/15 px-3 py-1.5 text-accent-soft hover:bg-accent/25"
          >
            Train it now
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-[280px_minmax(0,1fr)]">
          <div className="flex flex-col items-center gap-3">
            <div
              onDrop={onDrop}
              onDragOver={(e) => e.preventDefault()}
              className="flex h-56 w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border text-center text-xs text-zinc-500"
            >
              <UploadCloud size={22} />
              {fileName ? <span className="text-zinc-300">{fileName}</span> : "Drag an image here"}
              <label className="cursor-pointer rounded-md bg-white/5 px-2.5 py-1 text-zinc-300 hover:bg-white/10">
                Choose file
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleFile(file);
                  }}
                />
              </label>
            </div>
            <div className="text-center">
              <div className="mb-1 text-[11px] text-zinc-600">32×32 model input</div>
              <canvas ref={previewRef} width={32} height={32} className="h-24 w-24 rounded border border-border" />
            </div>
          </div>

          <div className="card p-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">Predictions</div>
            {error && <div className="text-xs text-rose-400">{error}</div>}
            {prediction ? (
              <ProbabilityBars labels={CLASS_LABELS} probabilities={prediction.probabilities} />
            ) : (
              <div className="text-xs text-zinc-600">Upload an image to classify it.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

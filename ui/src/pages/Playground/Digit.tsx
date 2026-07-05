import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eraser } from "lucide-react";
import { api, type ModelDescriptor, type Prediction } from "../../lib/tauri";
import ProbabilityBars from "../../components/ProbabilityBars";

const DIGIT_LABELS = Array.from({ length: 10 }, (_, i) => String(i));
const MODEL_IDS = ["mnist_mlp", "mnist_cnn", "mnist_attention"] as const;
type ModelId = (typeof MODEL_IDS)[number];

const PREDICTORS: Record<ModelId, (checkpoint: string, pixels: number[]) => Promise<Prediction>> = {
  mnist_mlp: api.predictMlp,
  mnist_cnn: api.predictCnn,
  mnist_attention: api.predictAttention,
};

export default function Digit() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);
  const navigate = useNavigate();

  const [models, setModels] = useState<Record<ModelId, ModelDescriptor | undefined>>({
    mnist_mlp: undefined,
    mnist_cnn: undefined,
    mnist_attention: undefined,
  });
  const [trained, setTrained] = useState<Record<ModelId, boolean>>({
    mnist_mlp: false,
    mnist_cnn: false,
    mnist_attention: false,
  });
  const [predictions, setPredictions] = useState<Record<ModelId, Prediction | undefined>>({
    mnist_mlp: undefined,
    mnist_cnn: undefined,
    mnist_attention: undefined,
  });
  const [errors, setErrors] = useState<Record<ModelId, string | undefined>>({
    mnist_mlp: undefined,
    mnist_cnn: undefined,
    mnist_attention: undefined,
  });

  useEffect(() => {
    api.listModels().then(async (all) => {
      const byId: Record<ModelId, ModelDescriptor | undefined> = {
        mnist_mlp: all.find((m) => m.id === "mnist_mlp"),
        mnist_cnn: all.find((m) => m.id === "mnist_cnn"),
        mnist_attention: all.find((m) => m.id === "mnist_attention"),
      };
      setModels(byId);
      const paths = MODEL_IDS.map((id) => byId[id]?.checkpoints[0] ?? "");
      const statuses = await api.checkpointStatus(paths);
      const trainedMap: Record<ModelId, boolean> = {
        mnist_mlp: statuses[0],
        mnist_cnn: statuses[1],
        mnist_attention: statuses[2],
      };
      setTrained(trainedMap);
    });
  }, []);

  useEffect(() => {
    clearCanvas();
  }, []);

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictions({ mnist_mlp: undefined, mnist_cnn: undefined, mnist_attention: undefined });
    setErrors({ mnist_mlp: undefined, mnist_cnn: undefined, mnist_attention: undefined });
  }

  function pos(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    drawing.current = true;
    lastPos.current = pos(e);
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!drawing.current) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const p = pos(e);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(lastPos.current!.x, lastPos.current!.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    lastPos.current = p;
  }

  function onPointerUp() {
    if (!drawing.current) return;
    drawing.current = false;
    lastPos.current = null;
    void predictAll();
  }

  function extractPixels(): number[] {
    const canvas = canvasRef.current!;
    const off = document.createElement("canvas");
    off.width = 28;
    off.height = 28;
    const octx = off.getContext("2d")!;
    octx.drawImage(canvas, 0, 0, 28, 28);
    const data = octx.getImageData(0, 0, 28, 28).data;
    const pixels: number[] = [];
    for (let i = 0; i < 28 * 28; i++) {
      pixels.push(data[i * 4] / 255);
    }
    return pixels;
  }

  async function predictAll() {
    const pixels = extractPixels();
    for (const id of MODEL_IDS) {
      const model = models[id];
      const checkpoint = model?.checkpoints[0];
      if (!model || !checkpoint || !trained[id]) continue;
      try {
        const checkpointPath = checkpoint;
        const prediction = await PREDICTORS[id](checkpointPath, pixels);
        setPredictions((prev) => ({ ...prev, [id]: prediction }));
        setErrors((prev) => ({ ...prev, [id]: undefined }));
      } catch (e) {
        setErrors((prev) => ({ ...prev, [id]: String(e) }));
      }
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-8 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-zinc-100">Draw a digit</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Draw 0-9 and compare how the MLP, CNN, and Attention models each read it.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        <div className="flex flex-col items-center gap-3">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            className="cursor-crosshair rounded-lg border border-border"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          />
          <button
            onClick={clearCanvas}
            className="flex items-center gap-1.5 rounded-lg bg-white/5 px-3 py-1.5 text-sm text-zinc-300 hover:bg-white/10"
          >
            <Eraser size={14} /> Clear
          </button>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {MODEL_IDS.map((id) => {
            const model = models[id];
            return (
              <div key={id} className="card p-3">
                <div className="mb-2 text-xs font-semibold text-zinc-300">{model?.display_name ?? id}</div>
                {!trained[id] ? (
                  <div className="flex flex-col gap-2 text-xs text-zinc-500">
                    Not trained yet.
                    <button
                      onClick={() => navigate(`/train/${id}`)}
                      className="w-fit rounded-md bg-accent/15 px-2 py-1 text-accent-soft hover:bg-accent/25"
                    >
                      Train this model
                    </button>
                  </div>
                ) : errors[id] ? (
                  <div className="text-xs text-rose-400">{errors[id]}</div>
                ) : predictions[id] ? (
                  <ProbabilityBars labels={DIGIT_LABELS} probabilities={predictions[id]!.probabilities} />
                ) : (
                  <div className="text-xs text-zinc-600">Draw a digit to predict.</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

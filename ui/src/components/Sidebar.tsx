import { NavLink } from "react-router-dom";
import { LayoutGrid, PlayCircle, History, FileJson, PenTool, ImageIcon, Brain } from "lucide-react";
import type { ReactNode } from "react";

function Item({ to, icon, label }: { to: string; icon: ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
          isActive
            ? "bg-accent/15 text-accent-soft"
            : "text-zinc-400 hover:bg-white/5 hover:text-zinc-100"
        }`
      }
    >
      {icon}
      {label}
    </NavLink>
  );
}

export default function Sidebar() {
  return (
    <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-surface/60 px-3 py-4">
      <div className="mb-6 flex items-center gap-2 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/20 text-accent-soft">
          <Brain size={18} />
        </div>
        <div>
          <div className="text-sm font-semibold text-zinc-100">Rust NN Studio</div>
          <div className="text-[11px] text-zinc-500">local training desk</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        <Item to="/" icon={<LayoutGrid size={16} />} label="Dashboard" />
        <Item to="/train" icon={<PlayCircle size={16} />} label="Train" />
        <Item to="/runs" icon={<History size={16} />} label="Runs" />
        <Item to="/configs" icon={<FileJson size={16} />} label="Configs" />
      </nav>

      <div className="mt-6 px-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
        Playground
      </div>
      <nav className="mt-1 flex flex-col gap-1">
        <Item to="/playground/digit" icon={<PenTool size={16} />} label="Draw a digit" />
        <Item to="/playground/cifar10" icon={<ImageIcon size={16} />} label="CIFAR-10 image" />
      </nav>

      <div className="mt-auto px-2 pt-4 text-[11px] leading-relaxed text-zinc-600">
        Trains &amp; runs the binaries in this repo directly — nothing leaves your machine.
      </div>
    </aside>
  );
}

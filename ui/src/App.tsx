import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Train from "./pages/Train";
import Runs from "./pages/Runs";
import Configs from "./pages/Configs";
import Digit from "./pages/Playground/Digit";
import Cifar10 from "./pages/Playground/Cifar10";

export default function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train/:modelId?" element={<Train />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/configs" element={<Configs />} />
          <Route path="/playground/digit" element={<Digit />} />
          <Route path="/playground/cifar10" element={<Cifar10 />} />
        </Routes>
      </main>
    </div>
  );
}

import { Link, Route, Routes } from "react-router-dom";
import DeploymentDetail from "./pages/DeploymentDetail";
import DeploymentList from "./pages/DeploymentList";

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          Inference Control Plane
        </Link>
        <span className="muted">vLLM on Kubernetes and Slurm</span>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<DeploymentList />} />
          <Route path="/deployments/:name" element={<DeploymentDetail />} />
        </Routes>
      </main>
    </div>
  );
}

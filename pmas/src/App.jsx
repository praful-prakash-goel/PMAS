import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import LoginPage from "./pages/Login";
import Layout from "./pages/Layout"
import HomePage from "./pages/Homepage";
import Machines from "./pages/Machines";
import Tickets from "./pages/Tickets";
import Technicians from "./pages/Technicians";
import Monitoring from "./pages/Monitoring";
import Analytics from "./pages/Analytics";

const AdminOnly = ({ role, children }) => {
  return role === "administrator" ? children : <Navigate to="/homepage" />;
  };

  const role = new URLSearchParams(window.location.search).get("user");

function App() {
  return (
    <Router>
      <Routes>

        <Route path="/" element={<LoginPage />} />

        {/* Everything else has navbar */}
        <Route element={<Layout />}>
          <Route path="/homepage" element={<HomePage />} />
          <Route path="/machines" element={<Machines />} />
          <Route path="/tickets" element={<Tickets />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/analytics"   element={<AdminOnly role={role}><Analytics /></AdminOnly>} />
          <Route path="/technicians" element={<AdminOnly role={role}><Technicians /></AdminOnly>} />
        </Route>

      </Routes>
    </Router>
  );
}

export default App;
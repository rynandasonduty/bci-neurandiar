import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import LiveSession from './pages/LiveSession';
import MonitorControl from './pages/MonitorControl';
import DataCenter from './pages/DataCenter';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-[#0f172a] font-['Inter']">
        {/* Sidebar Kiri */}
        <Sidebar />
        
        {/* Konten Utama (Diberi margin-left 64 karena lebar sidebar adalah w-64 atau 16rem) */}
        <div className="flex-1 ml-64 min-h-screen overflow-y-auto">
          <Routes>
            <Route path="/" element={<LiveSession />} />
            <Route path="/monitor" element={<MonitorControl />} />
            <Route path="/data" element={<DataCenter />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
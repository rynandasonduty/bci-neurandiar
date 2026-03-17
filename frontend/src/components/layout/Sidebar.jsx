import { NavLink } from 'react-router-dom';
import { Activity, LayoutDashboard, Database, BrainCircuit } from 'lucide-react';

const Sidebar = () => {
  const menuItems = [
    { name: 'Sesi Live BCI', path: '/', icon: <BrainCircuit size={20} /> },
    { name: 'Monitor & Kontrol', path: '/monitor', icon: <Activity size={20} /> },
    { name: 'Pusat Data & Evaluasi', path: '/data', icon: <Database size={20} /> },
  ];

  return (
    <div className="w-64 h-screen bg-slate-900 border-r border-slate-800 flex flex-col fixed left-0 top-0">
      {/* Logo Area */}
      <div className="h-20 flex items-center px-6 border-b border-slate-800">
        <h1 className="text-xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          NEURANDIAR
        </h1>
      </div>

      {/* Menu Links */}
      <div className="flex-1 py-6 flex flex-col gap-2 px-4">
        {menuItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 font-medium ${
                isActive
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20 shadow-[0_0_15px_rgba(37,99,235,0.1)]'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`
            }
          >
            {item.icon}
            {item.name}
          </NavLink>
        ))}
      </div>

      {/* Footer Info */}
      <div className="p-6 border-t border-slate-800 text-xs text-slate-500">
        <p>Proyek Skripsi BCI</p>
        <p className="mt-1">© 2026 Institut Teknologi Sepuluh Nopember</p>
      </div>
    </div>
  );
};

export default Sidebar;
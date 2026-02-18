import { useState } from "react";
import { useAllowedUsers } from "../hooks/useAllowedUsers.js";

interface AllowedUsersPanelProps {
  onClose: () => void;
}

export default function AllowedUsersPanel({ onClose }: AllowedUsersPanelProps) {
  const { allowedUsers, loading, error, addUser, removeUser } = useAllowedUsers();
  const [newLogin, setNewLogin] = useState("");

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = newLogin.trim();
    if (!trimmed) return;
    await addUser(trimmed);
    setNewLogin("");
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h2 className="text-lg font-semibold text-gray-100">
            Manage Access
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-100 transition-colors"
          >
            &times;
          </button>
        </div>

        <div className="p-4">
          {error && (
            <div className="mb-3 px-3 py-2 rounded bg-red-900/50 border border-red-700 text-red-200 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleAdd} className="flex gap-2 mb-4">
            <input
              type="text"
              value={newLogin}
              onChange={(e) => setNewLogin(e.target.value)}
              placeholder="GitHub username"
              className="flex-1 px-3 py-1.5 rounded bg-gray-800 border border-gray-600 text-gray-100 text-sm placeholder-gray-500 focus:outline-none focus:border-gray-400"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded bg-blue-600 text-white text-sm hover:bg-blue-500 transition-colors"
            >
              Add
            </button>
          </form>

          {loading ? (
            <p className="text-sm text-gray-400">Loading...</p>
          ) : allowedUsers.length === 0 ? (
            <p className="text-sm text-gray-400">
              No users on the whitelist.
            </p>
          ) : (
            <ul className="space-y-1 max-h-60 overflow-y-auto">
              {allowedUsers.map((u) => (
                <li
                  key={u.id}
                  className="flex items-center justify-between px-3 py-2 rounded bg-gray-800"
                >
                  <span className="text-sm text-gray-100">{u.githubLogin}</span>
                  <button
                    onClick={() => removeUser(u.githubLogin)}
                    className="text-xs text-gray-400 hover:text-red-400 transition-colors"
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

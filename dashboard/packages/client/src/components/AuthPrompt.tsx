import { useState } from "react";
import { useAuthStore } from "../hooks/useAuth.js";

export default function AuthPrompt() {
  const { showPrompt, setToken, dismissPrompt } = useAuthStore();
  const [value, setValue] = useState("");

  if (!showPrompt) return null;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim()) {
      setToken(value.trim());
      setValue("");
    }
  }

  function handleCancel() {
    setValue("");
    dismissPrompt();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <form
        onSubmit={handleSubmit}
        className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-full max-w-sm shadow-xl"
      >
        <h2 className="text-lg font-semibold text-gray-100 mb-1">
          Authentication Required
        </h2>
        <p className="text-sm text-gray-400 mb-4">
          Enter the dashboard auth token to continue.
        </p>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Auth token"
          autoFocus
          className="w-full px-3 py-2 rounded bg-gray-800 border border-gray-600 text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500 mb-4"
        />
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={handleCancel}
            className="px-4 py-2 text-sm rounded text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-500 transition-colors"
          >
            Submit
          </button>
        </div>
      </form>
    </div>
  );
}

import { useReports, useCreateReport, useDeleteReport } from "../../hooks/useReports.js";
import { useExperimentStore } from "../../stores/experimentStore.js";

export default function ReportsSidebar() {
  const { selectedReportId, setSelectedReportId, sidebarOpen, setSidebarOpen } =
    useExperimentStore();
  const { data, isLoading } = useReports();
  const createReport = useCreateReport();
  const deleteReport = useDeleteReport();

  const reports = data?.reports ?? [];

  const handleNew = async () => {
    const result = await createReport.mutateAsync(undefined);
    setSelectedReportId(result.report.id);
    setSidebarOpen(false);
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedReportId === id) setSelectedReportId(null);
    await deleteReport.mutateAsync(id);
  };

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`
          fixed md:static z-30 top-0 left-0 h-full md:h-auto
          w-64 bg-gray-950 border-r border-gray-800
          flex flex-col transition-transform duration-200
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"}
          md:translate-x-0
        `}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Reports</h2>
          <button
            className="md:hidden text-gray-400 hover:text-gray-200"
            onClick={() => setSidebarOpen(false)}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New report button */}
        <div className="px-3 py-2 border-b border-gray-800">
          <button
            onClick={handleNew}
            disabled={createReport.isPending}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded transition-colors bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white border border-gray-700"
          >
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
            New Report
          </button>
        </div>

        {/* Report list */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="px-4 py-3 text-sm text-gray-500">Loading...</div>
          ) : reports.length === 0 ? (
            <div className="px-4 py-3 text-sm text-gray-500">
              No reports yet
            </div>
          ) : (
            <ul className="py-1">
              {reports.map((report) => (
                <li key={report.id}>
                  <button
                    onClick={() => {
                      setSelectedReportId(report.id);
                      setSidebarOpen(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 transition-colors group ${
                      selectedReportId === report.id
                        ? "bg-gray-800 border-l-2 border-blue-500"
                        : "hover:bg-gray-900 border-l-2 border-transparent"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-200 truncate">
                        {report.title}
                      </span>
                      <button
                        onClick={(e) => handleDelete(report.id, e)}
                        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 transition-opacity p-0.5"
                        title="Delete report"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      {new Date(report.updatedAt).toLocaleDateString()}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}

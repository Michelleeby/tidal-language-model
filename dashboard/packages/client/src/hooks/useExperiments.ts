import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: () => api.getExperiments(),
  });
}

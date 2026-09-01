interface Window {
  cagent: {
    send(command: Record<string, unknown>): void;
    onEvent(listener: (event: BackendEvent) => void): () => void;
    chooseWorkspace(): Promise<string | null>;
    minimize(): void;
    maximize(): void;
    close(): void;
  };
}

interface BackendEvent {
  type: string;
  [key: string]: unknown;
}


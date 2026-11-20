import type { DraftFile, DraftStorage } from "./types.ts";

const DEFAULT_KEY = "speech-review-draft-v1";

export class LocalStorageDraftStore implements DraftStorage {
  private readonly storage: Storage;
  private readonly key: string;

  constructor(storage: Storage, key = DEFAULT_KEY) {
    this.storage = storage;
    this.key = key;
  }

  load(): unknown | null {
    const raw = this.storage.getItem(this.key);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw) as unknown;
    } catch {
      return null;
    }
  }

  save(draft: DraftFile): void {
    this.storage.setItem(this.key, JSON.stringify(draft));
  }

  clear(): void {
    this.storage.removeItem(this.key);
  }
}

export class MemoryDraftStore implements DraftStorage {
  private draft: DraftFile | null = null;

  load(): unknown | null {
    return this.draft ? structuredClone(this.draft) : null;
  }

  save(draft: DraftFile): void {
    this.draft = structuredClone(draft);
  }

  clear(): void {
    this.draft = null;
  }
}

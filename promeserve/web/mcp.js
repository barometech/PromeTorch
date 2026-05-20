/* ============================================================
   PromeTorch — MCP client (contributor:F2)
   ============================================================
   Backend MCP endpoints (commit 7d48408 / 0428eeb):
     GET  /api/mcp/tools     — список tools (8+ builtin + MCP-discovered)
     GET  /api/mcp/servers   — список подключённых MCP server'ов
     POST /api/mcp/call      — выполнить tool { name, arguments }

   Config: ~/.promeserve/mcp.json (Cline-compatible)
   Формат: { "mcpServers": { "name": { "command": "...", "args": [], "env": {} } } }

   F2 UI код в index.html уже:
     - хранит state.mcpServers, presets, enabled/disabled state
     - рисует sidebar list + management modal
     - готов wire'ить через PromeMcp ниже
   ============================================================ */

(function() {
  'use strict';

  const API_BASE = (typeof window !== 'undefined' ? window.location.origin : '');

  const PromeMcp = {
    /** Cached MCP catalog */
    _cache: { tools: null, servers: null, ts: 0 },
    CACHE_TTL_MS: 15000,

    /** List all tools (built-in + discovered from connected MCP servers) */
    async listTools(force = false) {
      const now = Date.now();
      if (!force && this._cache.tools && (now - this._cache.ts) < this.CACHE_TTL_MS) {
        return this._cache.tools;
      }
      try {
        const res = await fetch(`${API_BASE}/api/mcp/tools`);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        this._cache.tools = data.tools || data || [];
        this._cache.ts = now;
        return this._cache.tools;
      } catch (e) {
        console.warn('[MCP] listTools failed', e);
        return [];
      }
    },

    /** List connected MCP servers (from ~/.promeserve/mcp.json) */
    async listServers(force = false) {
      const now = Date.now();
      if (!force && this._cache.servers && (now - this._cache.ts) < this.CACHE_TTL_MS) {
        return this._cache.servers;
      }
      try {
        const res = await fetch(`${API_BASE}/api/mcp/servers`);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        this._cache.servers = data.servers || data || [];
        return this._cache.servers;
      } catch (e) {
        console.warn('[MCP] listServers failed', e);
        return [];
      }
    },

    /** Invoke a tool by name with arguments object */
    async callTool(name, args) {
      try {
        const res = await fetch(`${API_BASE}/api/mcp/call`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, arguments: args || {} }),
        });
        if (!res.ok) {
          const err = await res.text();
          return { error: `HTTP ${res.status}: ${err}` };
        }
        return await res.json();
      } catch (e) {
        return { error: String(e.message || e) };
      }
    },

    /** Invalidate cache (call after server connect/disconnect) */
    invalidate() {
      this._cache = { tools: null, servers: null, ts: 0 };
    },

    /** Preset templates — для UI "Add MCP server" (copy into mcp.json) */
    PRESETS: [
      { id: 'filesystem', name: 'Filesystem', desc: 'Local FS read/write/list', command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-filesystem', '<root-path>'] },
      { id: 'github', name: 'GitHub', desc: 'GitHub Issues/Pulls/Code Search', command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-github'],
        env: { GITHUB_PERSONAL_ACCESS_TOKEN: '<token>' } },
      { id: 'fetch', name: 'Fetch (web)', desc: 'HTTP+HTML→markdown', command: 'uvx',
        args: ['mcp-server-fetch'] },
      { id: 'brave-search', name: 'Brave Search', desc: 'Web search via Brave API', command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-brave-search'],
        env: { BRAVE_API_KEY: '<key>' } },
      { id: 'memory', name: 'Memory', desc: 'Knowledge graph persistent memory', command: 'npx',
        args: ['-y', '@modelcontextprotocol/server-memory'] },
      { id: 'sequential-thinking', name: 'Sequential Thinking', desc: 'Step-by-step decomposition',
        command: 'npx', args: ['-y', '@modelcontextprotocol/server-sequential-thinking'] },
    ],

    /** Generate mcp.json snippet for a preset (for user to paste into config) */
    presetToConfig(presetId, options = {}) {
      const p = this.PRESETS.find(x => x.id === presetId);
      if (!p) return null;
      const cfg = { command: p.command, args: [...p.args] };
      if (p.env) cfg.env = { ...p.env, ...(options.env || {}) };
      return cfg;
    },
  };

  if (typeof window !== 'undefined') window.PromeMcp = PromeMcp;
})();

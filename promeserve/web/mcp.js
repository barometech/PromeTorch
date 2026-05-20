/* ============================================================
   PromeTorch — MCP client stub (contributor:F2)
   ============================================================
   Этот модуль — заготовка под backend MCP-интеграцию.
   Сейчас экспортирует API для подключения к JSON-RPC 2.0 over
   Streamable HTTP MCP endpoint (backend будет на /api/mcp).

   F2 UI код в index.html уже:
     - хранит state.mcpServers, presets, enabled/disabled state
     - рисует sidebar list + management modal
     - подписан на toggle/add/remove события

   Когда backend появится (см. docs/research/AGENT_STACK_R2.md):
     - заменить PromeMcp.connect/listTools/callTool на реальные
       fetch к /api/mcp с JSON-RPC payload
     - state.mcpServers.tools будет приходить из tools/list
     - tool_call rendering уже совместим (Hermes <tool_call> тег)
   ============================================================ */

(function() {
  'use strict';

  const PromeMcp = {
    /** Initialize MCP connection (placeholder) */
    async connect(server) {
      // TODO(backend): POST /api/mcp/connect with { command, args, env }
      console.log('[MCP stub] connect', server);
      return { ok: true, sessionId: 'mock-' + Math.random().toString(36).slice(2, 8) };
    },

    /** List tools available on a connected server */
    async listTools(sessionId) {
      // TODO(backend): JSON-RPC tools/list to /api/mcp/{sessionId}
      console.log('[MCP stub] listTools', sessionId);
      return [];
    },

    /** Invoke a tool */
    async callTool(sessionId, toolName, args) {
      // TODO(backend): JSON-RPC tools/call to /api/mcp/{sessionId}
      console.log('[MCP stub] callTool', toolName, args);
      return { content: [{ type: 'text', text: 'MCP backend not connected yet.' }] };
    },

    /** Disconnect a server */
    async disconnect(sessionId) {
      // TODO(backend): DELETE /api/mcp/{sessionId}
      console.log('[MCP stub] disconnect', sessionId);
      return { ok: true };
    },

    /** List of preset MCP servers (kept in sync with index.html MCP_PRESETS) */
    PRESETS: [
      { id: 'filesystem', name: 'Filesystem', desc: 'Local FS read/write/list', npm: '@modelcontextprotocol/server-filesystem' },
      { id: 'github',     name: 'GitHub',     desc: 'GitHub API integration', npm: '@modelcontextprotocol/server-github' },
      { id: 'fetch',      name: 'Fetch',      desc: 'HTTP+HTML→markdown',     npm: 'mcp-server-fetch (uvx)' },
      { id: 'brave-search', name: 'Brave Search', desc: 'Web search via Brave API', npm: '@modelcontextprotocol/server-brave-search' },
      { id: 'memory',     name: 'Memory',     desc: 'Knowledge graph store',   npm: '@modelcontextprotocol/server-memory' },
      { id: 'sequential-thinking', name: 'Sequential Thinking', desc: 'Step-by-step decomposition', npm: '@modelcontextprotocol/server-sequential-thinking' },
    ],
  };

  // Expose globally for index.html
  if (typeof window !== 'undefined') window.PromeMcp = PromeMcp;
})();

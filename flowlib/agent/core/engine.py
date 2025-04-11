from typing import Optional, Dict, Any
from flowlib.agent.core.agent import AgentCore
from flowlib.agent.memory.agent_memory import AgentMemory
from flowlib.agent.core.agent import AgentConfig

class AgentEngine:
    """Engine for managing agent execution and memory"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent = AgentCore(config)
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize the agent's memory system"""
        memory_config = self.config.memory_config or {}
        self.agent._memory = AgentMemory(**memory_config)
        
        # Ensure memory is properly initialized
        if not self.agent._memory:
            raise RuntimeError("Failed to initialize agent memory system")
            
        # Register memory with agent
        self.agent.register_memory(self.agent._memory)
    
    async def initialize(self):
        """Initialize the agent engine"""
        await self.agent.initialize()
        await self.agent._memory.initialize()
    
    async def shutdown(self):
        """Shutdown the agent engine"""
        await self.agent._memory.shutdown()
        await self.agent.shutdown() 
import axios from 'axios'

export default {
  namespaced: true,
  
  state: {
    gameState: null,
    playerStats: null,
    inventory: null,
    currentTask: null,
    battleStatus: null
  },

  mutations: {
    setGameState(state, gameState) {
      state.gameState = gameState
    },
    setPlayerStats(state, stats) {
      state.playerStats = stats
    },
    setInventory(state, inventory) {
      state.inventory = inventory
    },
    setCurrentTask(state, task) {
      state.currentTask = task
    },
    setBattleStatus(state, status) {
      state.battleStatus = status
    }
  },

  actions: {
    async updateGameState({ commit }, gameState) {
      commit('setGameState', gameState)
      if (gameState.playerStats) {
        commit('setPlayerStats', gameState.playerStats)
      }
      if (gameState.inventory) {
        commit('setInventory', gameState.inventory)
      }
      if (gameState.currentTask) {
        commit('setCurrentTask', gameState.currentTask)
      }
      if (gameState.battleStatus) {
        commit('setBattleStatus', gameState.battleStatus)
      }
    },

    async startBattle({ commit }) {
      try {
        const response = await axios.post('/api/game/battle/start')
        commit('setBattleStatus', response.data.status)
        return response.data
      } catch (error) {
        console.error('Failed to start battle:', error)
        throw error
      }
    },

    async useItem({ commit, state }, itemId) {
      try {
        const response = await axios.post('/api/game/inventory/use', { itemId })
        const updatedInventory = state.inventory.filter(item => item.id !== itemId)
        commit('setInventory', updatedInventory)
        return response.data
      } catch (error) {
        console.error('Failed to use item:', error)
        throw error
      }
    },

    async completeTask({ commit }, taskId) {
      try {
        const response = await axios.post('/api/game/task/complete', { taskId })
        commit('setCurrentTask', null)
        return response.data
      } catch (error) {
        console.error('Failed to complete task:', error)
        throw error
      }
    }
  },

  getters: {
    isInBattle: state => state.battleStatus !== null,
    currentHealth: state => state.playerStats?.health || 0,
    currentMana: state => state.playerStats?.mana || 0,
    inventoryItems: state => state.inventory || [],
    activeTask: state => state.currentTask,
    gameStatus: state => state.gameState?.status
  }
}

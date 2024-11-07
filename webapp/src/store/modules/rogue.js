import axios from 'axios'

const state = {
  gameState: {
    level: 0,
    health: 100,
    gold: 0,
    blessings: []
  },
  isRunning: false,
  error: null
}

const mutations = {
  setGameState(state, gameState) {
    state.gameState = gameState
  },
  setIsRunning(state, isRunning) {
    state.isRunning = isRunning
  },
  setError(state, error) {
    state.error = error
  }
}

const actions = {
  async startNewRun({ commit }) {
    try {
      const response = await axios.post('/api/rogue/start')
      commit('setGameState', response.data)
      commit('setIsRunning', true)
      commit('setError', null)
    } catch (error) {
      commit('setError', error.response.data.error || 'Failed to start new run')
      console.error('Failed to start new run:', error)
    }
  },
  async advanceLevel({ commit }) {
    try {
      const response = await axios.post('/api/rogue/advance')
      commit('setGameState', response.data)
      commit('setError', null)
    } catch (error) {
      commit('setError', error.response.data.error || 'Failed to advance level')
      console.error('Failed to advance level:', error)
    }
  },
  async takeDamage({ commit }, amount) {
    try {
      const response = await axios.post('/api/rogue/take_damage', { amount })
      commit('setGameState', response.data)
      commit('setError', null)
    } catch (error) {
      commit('setError', error.response.data.error || 'Failed to take damage')
      console.error('Failed to take damage:', error)
    }
  },
  async endRun({ commit }) {
    try {
      const response = await axios.post('/api/rogue/end_run')
      commit('setGameState', response.data.final_state)
      commit('setIsRunning', false)
      commit('setError', null)
    } catch (error) {
      commit('setError', error.response.data.error || 'Failed to end run')
      console.error('Failed to end run:', error)
    }
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
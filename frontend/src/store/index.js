import { createStore } from 'vuex'
import axios from 'axios'
import game from './modules/game'
import rogue from './modules/rogue'

export default createStore({
  state: {
    status: null,
    loading: false,
    error: null
  },
  mutations: {
    setStatus(state, status) {
      state.status = status
    },
    setLoading(state, loading) {
      state.loading = loading
    },
    setError(state, error) {
      state.error = error
    }
  },
  actions: {
    async startAutomation({ commit }) {
      commit('setLoading', true)
      try {
        const response = await axios.post('/api/start')
        commit('setStatus', response.data.status)
      } catch (error) {
        commit('setError', error.message)
        console.error('Failed to start automation:', error)
      } finally {
        commit('setLoading', false)
      }
    },
    
    async stopAutomation({ commit }) {
      commit('setLoading', true)
      try {
        const response = await axios.post('/api/stop')
        commit('setStatus', response.data.status)
      } catch (error) {
        commit('setError', error.message)
        console.error('Failed to stop automation:', error)
      } finally {
        commit('setLoading', false)
      }
    },
    
    async fetchGameState({ commit, dispatch }) {
      commit('setLoading', true)
      try {
        const response = await axios.get('/api/game-state')
        dispatch('game/updateGameState', response.data)
        dispatch('rogue/updateRogueState', response.data)
      } catch (error) {
        commit('setError', error.message)
        console.error('Failed to fetch game state:', error)
      } finally {
        commit('setLoading', false)
      }
    },
    
    clearError({ commit }) {
      commit('setError', null)
    }
  },
  modules: {
    game,
    rogue
  },
  getters: {
    isLoading: state => state.loading,
    hasError: state => state.error !== null,
    errorMessage: state => state.error,
    automationStatus: state => state.status
  }
})

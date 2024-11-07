import { createStore } from 'vuex'
import axios from 'axios'
import game from './modules/game'

export default createStore({
  state: {
    status: null
  },
  mutations: {
    setStatus(state, status) {
      state.status = status
    }
  },
  actions: {
    async startAutomation({ commit }) {
      try {
        const response = await axios.post('/api/start')
        commit('setStatus', response.data.status)
      } catch (error) {
        console.error('Failed to start automation:', error)
      }
    },
    async stopAutomation({ commit }) {
      try {
        const response = await axios.post('/api/stop')
        commit('setStatus', response.data.status)
      } catch (error) {
        console.error('Failed to stop automation:', error)
      }
    },
    async fetchGameState({ dispatch }) {
      try {
        const response = await axios.get('/api/game-state')
        dispatch('game/updateGameState', response.data)
      } catch (error) {
        console.error('Failed to fetch game state:', error)
      }
    }
  },
  modules: {
    game
  }
})
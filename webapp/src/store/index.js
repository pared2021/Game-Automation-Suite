import { createStore } from 'vuex'
import axios from 'axios'
import rogue from './modules/rogue'

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
    }
  },
  modules: {
    rogue
  }
})
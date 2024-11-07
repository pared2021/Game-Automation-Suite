const state = {
  level: 0,
  activeBlessings: []
}

const mutations = {
  SET_LEVEL(state, level) {
    state.level = level
  },
  SET_ACTIVE_BLESSINGS(state, blessings) {
    state.activeBlessings = blessings
  }
}

const actions = {
  updateGameState({ commit }, { level, activeBlessings }) {
    commit('SET_LEVEL', level)
    commit('SET_ACTIVE_BLESSINGS', activeBlessings)
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions
}
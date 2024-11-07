import axios from 'axios'

export default {
  namespaced: true,
  
  state: {
    rogueState: null,
    currentRun: null,
    availableCharacters: [],
    selectedCharacter: null,
    dungeonLevel: 1,
    rewards: [],
    abilities: [],
    artifacts: []
  },

  mutations: {
    setRogueState(state, rogueState) {
      state.rogueState = rogueState
    },
    setCurrentRun(state, run) {
      state.currentRun = run
    },
    setAvailableCharacters(state, characters) {
      state.availableCharacters = characters
    },
    setSelectedCharacter(state, character) {
      state.selectedCharacter = character
    },
    setDungeonLevel(state, level) {
      state.dungeonLevel = level
    },
    setRewards(state, rewards) {
      state.rewards = rewards
    },
    setAbilities(state, abilities) {
      state.abilities = abilities
    },
    setArtifacts(state, artifacts) {
      state.artifacts = artifacts
    },
    addArtifact(state, artifact) {
      state.artifacts.push(artifact)
    },
    removeArtifact(state, artifactId) {
      state.artifacts = state.artifacts.filter(a => a.id !== artifactId)
    },
    addAbility(state, ability) {
      state.abilities.push(ability)
    }
  },

  actions: {
    async updateRogueState({ commit }, gameState) {
      if (gameState.rogueMode) {
        commit('setRogueState', gameState.rogueMode)
        if (gameState.rogueMode.currentRun) {
          commit('setCurrentRun', gameState.rogueMode.currentRun)
        }
        if (gameState.rogueMode.dungeonLevel) {
          commit('setDungeonLevel', gameState.rogueMode.dungeonLevel)
        }
      }
    },

    async startRogueRun({ commit }, characterId) {
      try {
        const response = await axios.post('/api/rogue/start', { characterId })
        commit('setCurrentRun', response.data.run)
        commit('setSelectedCharacter', response.data.character)
        commit('setDungeonLevel', 1)
        return response.data
      } catch (error) {
        console.error('Failed to start rogue run:', error)
        throw error
      }
    },

    async endRogueRun({ commit }) {
      try {
        const response = await axios.post('/api/rogue/end')
        commit('setCurrentRun', null)
        commit('setRewards', response.data.rewards)
        return response.data
      } catch (error) {
        console.error('Failed to end rogue run:', error)
        throw error
      }
    },

    async selectCharacter({ commit }, character) {
      try {
        const response = await axios.post('/api/rogue/character/select', { characterId: character.id })
        commit('setSelectedCharacter', response.data.character)
        return response.data
      } catch (error) {
        console.error('Failed to select character:', error)
        throw error
      }
    },

    async proceedToNextLevel({ commit, state }) {
      try {
        const response = await axios.post('/api/rogue/level/proceed')
        commit('setDungeonLevel', state.dungeonLevel + 1)
        return response.data
      } catch (error) {
        console.error('Failed to proceed to next level:', error)
        throw error
      }
    },

    async collectArtifact({ commit }, artifactId) {
      try {
        const response = await axios.post('/api/rogue/artifact/collect', { artifactId })
        commit('addArtifact', response.data.artifact)
        return response.data
      } catch (error) {
        console.error('Failed to collect artifact:', error)
        throw error
      }
    },

    async unlockAbility({ commit }, abilityId) {
      try {
        const response = await axios.post('/api/rogue/ability/unlock', { abilityId })
        commit('addAbility', response.data.ability)
        return response.data
      } catch (error) {
        console.error('Failed to unlock ability:', error)
        throw error
      }
    }
  },

  getters: {
    isInRogueRun: state => state.currentRun !== null,
    currentCharacter: state => state.selectedCharacter,
    currentLevel: state => state.dungeonLevel,
    currentArtifacts: state => state.artifacts,
    currentAbilities: state => state.abilities,
    availableCharactersList: state => state.availableCharacters,
    currentRewards: state => state.rewards,
    rogueProgress: state => {
      if (!state.currentRun) return 0
      return (state.dungeonLevel - 1) * 20 // 假设每层代表20%的进度
    }
  }
}

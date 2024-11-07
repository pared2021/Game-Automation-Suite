<template>
  <div class="rogue">
    <h1>{{ $t('rogue.title') }}</h1>

    <!-- 当前运行状态 -->
    <div class="run-status" v-if="isInRogueRun">
      <div class="level-info">
        <h2>{{ $t('rogue.level', { level: currentLevel }) }}</h2>
        <div class="progress-container">
          <div class="progress-bar">
            <div :style="{ width: `${rogueProgress}%` }" class="progress"></div>
          </div>
          <span>{{ rogueProgress }}%</span>
        </div>
      </div>

      <!-- 当前角色信息 -->
      <div class="character-info" v-if="currentCharacter">
        <img :src="currentCharacter.avatar" :alt="currentCharacter.name" class="character-avatar">
        <div class="character-details">
          <h3>{{ currentCharacter.name }}</h3>
          <p>{{ currentCharacter.class }}</p>
        </div>
      </div>

      <!-- 神器和能力 -->
      <div class="powers-grid">
        <div class="artifacts">
          <h3>{{ $t('rogue.artifacts') }}</h3>
          <div class="artifacts-list">
            <div v-for="artifact in currentArtifacts" :key="artifact.id" class="artifact-item">
              <img :src="artifact.icon" :alt="artifact.name">
              <span>{{ artifact.name }}</span>
            </div>
          </div>
        </div>

        <div class="abilities">
          <h3>{{ $t('rogue.abilities') }}</h3>
          <div class="abilities-list">
            <div v-for="ability in currentAbilities" :key="ability.id" class="ability-item">
              <img :src="ability.icon" :alt="ability.name">
              <span>{{ ability.name }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 控制按钮 -->
      <div class="controls">
        <button @click="proceedToNextLevel" :disabled="isLoading" class="proceed-btn">
          {{ $t('rogue.proceed') }}
        </button>
        <button @click="endRogueRun" :disabled="isLoading" class="end-run-btn">
          {{ $t('rogue.endRun') }}
        </button>
      </div>
    </div>

    <!-- 角色选择 -->
    <div v-else class="character-selection">
      <h2>{{ $t('rogue.selectCharacter') }}</h2>
      <div class="characters-grid">
        <div 
          v-for="character in availableCharactersList" 
          :key="character.id"
          class="character-card"
          :class="{ selected: selectedCharacter?.id === character.id }"
          @click="selectCharacter(character)"
        >
          <img :src="character.avatar" :alt="character.name">
          <h3>{{ character.name }}</h3>
          <p>{{ character.class }}</p>
          <div class="character-stats">
            <span>{{ $t('rogue.health') }}: {{ character.health }}</span>
            <span>{{ $t('rogue.attack') }}: {{ character.attack }}</span>
          </div>
        </div>
      </div>

      <button 
        @click="startRun" 
        :disabled="!selectedCharacter || isLoading" 
        class="start-run-btn"
      >
        {{ $t('rogue.startRun') }}
      </button>
    </div>

    <!-- 奖励展示 -->
    <div v-if="currentRewards.length > 0" class="rewards-overlay">
      <div class="rewards-content">
        <h2>{{ $t('rogue.rewards') }}</h2>
        <div class="rewards-grid">
          <div v-for="reward in currentRewards" :key="reward.id" class="reward-item">
            <img :src="reward.icon" :alt="reward.name">
            <h4>{{ reward.name }}</h4>
            <p>{{ reward.description }}</p>
          </div>
        </div>
        <button @click="closeRewards" class="close-rewards-btn">
          {{ $t('rogue.continue') }}
        </button>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
    </div>
  </div>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  name: 'Rogue',

  data() {
    return {
      selectedCharacter: null
    }
  },

  computed: {
    ...mapGetters([
      'isLoading'
    ]),
    ...mapGetters('rogue', [
      'isInRogueRun',
      'currentCharacter',
      'currentLevel',
      'currentArtifacts',
      'currentAbilities',
      'availableCharactersList',
      'currentRewards',
      'rogueProgress'
    ])
  },

  methods: {
    ...mapActions('rogue', [
      'startRogueRun',
      'endRogueRun',
      'selectCharacter',
      'proceedToNextLevel'
    ]),

    async startRun() {
      if (!this.selectedCharacter) return
      try {
        await this.startRogueRun(this.selectedCharacter.id)
      } catch (error) {
        console.error('Failed to start run:', error)
      }
    },

    closeRewards() {
      // 清除奖励显示的逻辑
      this.$store.commit('rogue/setRewards', [])
    }
  }
}
</script>

<style scoped>
.rogue {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.run-status {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  margin-bottom: 30px;
}

.level-info {
  text-align: center;
  margin-bottom: 20px;
}

.progress-container {
  max-width: 400px;
  margin: 0 auto;
}

.progress-bar {
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
  margin: 10px 0;
}

.progress {
  height: 100%;
  background: #42b983;
  transition: width 0.3s ease;
}

.character-info {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 20px;
}

.character-avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  object-fit: cover;
}

.powers-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 20px 0;
}

.artifacts-list, .abilities-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 10px;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 8px;
}

.artifact-item, .ability-item {
  text-align: center;
  padding: 10px;
  background: white;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.artifact-item img, .ability-item img {
  width: 40px;
  height: 40px;
  margin-bottom: 5px;
}

.controls {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-top: 20px;
}

.proceed-btn, .end-run-btn, .start-run-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
}

.proceed-btn {
  background: #42b983;
  color: white;
}

.end-run-btn {
  background: #dc3545;
  color: white;
}

.character-selection {
  text-align: center;
}

.characters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.character-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  cursor: pointer;
  transition: all 0.3s ease;
}

.character-card:hover {
  transform: translateY(-5px);
}

.character-card.selected {
  border: 2px solid #42b983;
}

.character-card img {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  margin-bottom: 10px;
}

.character-stats {
  display: flex;
  justify-content: space-around;
  margin-top: 10px;
}

.rewards-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.rewards-content {
  background: white;
  padding: 30px;
  border-radius: 12px;
  max-width: 800px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
}

.rewards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.reward-item {
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255,255,255,0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #42b983;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .powers-grid {
    grid-template-columns: 1fr;
  }

  .characters-grid {
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  }
}
</style>

<template>
  <div class="home">
    <h1>{{ $t('general.title') }}</h1>
    <button @click="startAutomation">{{ $t('general.start') }}</button>
    <button @click="stopAutomation">{{ $t('general.stop') }}</button>
    <div v-if="status">
      <h2>{{ $t('general.status') }}: {{ status }}</h2>
    </div>
    <GameStatus />
  </div>
</template>

<script>
import { mapActions, mapState } from 'vuex'
import GameStatus from '@/components/GameStatus.vue'

export default {
  name: 'Home',
  components: {
    GameStatus
  },
  computed: {
    ...mapState(['status'])
  },
  methods: {
    ...mapActions(['startAutomation', 'stopAutomation'])
  }
}
</script>
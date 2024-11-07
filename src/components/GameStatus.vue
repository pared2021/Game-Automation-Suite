<template>
  <div class="game-status">
    <h2>{{ $t('game.status') }}</h2>
    <p>{{ $t('game.currentLevel') }}: {{ level }}</p>
    <p>{{ $t('game.activeBlessings') }}: {{ activeBlessings.join(', ') }}</p>
  </div>
</template>

<script>
import { mapState } from 'vuex'

export default {
  name: 'GameStatus',
  computed: {
    ...mapState({
      level: state => state.game.level,
      activeBlessings: state => state.game.activeBlessings
    })
  }
}
</script>
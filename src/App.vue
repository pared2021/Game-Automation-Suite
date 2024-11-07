<template>
  <div id="app">
    <nav>
      <router-link to="/">{{ $t('general.home') }}</router-link> |
      <router-link to="/about">{{ $t('general.about') }}</router-link>
    </nav>
    <select v-model="$i18n.locale">
      <option value="en-US">English</option>
      <option value="zh-CN">中文</option>
      <option value="ja-JP">日本語</option>
    </select>
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    // 在组件挂载后获取游戏状态
    this.$store.dispatch('fetchGameState')
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

nav {
  padding: 30px;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
}

nav a.router-link-exact-active {
  color: #42b983;
}

select {
  margin-bottom: 20px;
}
</style>
import { createI18n } from 'vue-i18n'
import enUS from '../config/i18n/en-US.json'
import zhCN from '../config/i18n/zh-CN.json'
import jaJP from '../config/i18n/ja-JP.json'

export const i18n = createI18n({
  legacy: false,
  locale: 'en-US', // 设置默认语言
  fallbackLocale: 'en-US', // 设置回退语言
  messages: {
    'en-US': enUS,
    'zh-CN': zhCN,
    'ja-JP': jaJP
  }
})
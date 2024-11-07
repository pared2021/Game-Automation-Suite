import { createI18n } from 'vue-i18n'

const messages = {
  'en-US': {
    general: {
      home: 'Home',
      about: 'About',
      game: 'Game',
      rogue: 'Rogue Mode'
    },
    home: {
      title: 'Game Automation Suite',
      status: 'Status',
      start: 'Start Automation',
      stop: 'Stop Automation',
      playerStats: 'Player Stats',
      health: 'Health',
      mana: 'Mana',
      currentTask: 'Current Task',
      progress: 'Progress'
    },
    game: {
      title: 'Game Control',
      playerStats: 'Player Statistics',
      health: 'Health',
      mana: 'Mana',
      inventory: 'Inventory',
      currentTask: 'Current Task',
      completeTask: 'Complete Task',
      battle: 'Battle',
      startBattle: 'Start Battle'
    },
    rogue: {
      title: 'Rogue Mode',
      level: 'Level {level}',
      artifacts: 'Artifacts',
      abilities: 'Abilities',
      proceed: 'Proceed to Next Level',
      endRun: 'End Run',
      selectCharacter: 'Select Character',
      startRun: 'Start Run',
      health: 'Health',
      attack: 'Attack',
      rewards: 'Rewards',
      continue: 'Continue'
    },
    about: {
      title: 'About',
      projectInfo: 'Project Information',
      description: 'Game Automation Suite is an advanced automation tool powered by AI for game automation and optimization.',
      version: 'Version',
      features: 'Features',
      gameAutomation: 'Game Automation',
      gameAutomationDesc: 'Intelligent automation of game tasks and battles',
      aiDecision: 'AI Decision Making',
      aiDecisionDesc: 'Advanced AI algorithms for optimal decision making',
      rogueMode: 'Rogue Mode',
      rogueModeDesc: 'Procedurally generated challenges with unique rewards',
      dataAnalysis: 'Data Analysis',
      dataAnalysisDesc: 'Comprehensive analysis of game performance and statistics',
      techStack: 'Technology Stack',
      contributors: 'Contributors'
    }
  },
  'zh-CN': {
    general: {
      home: '首页',
      about: '关于',
      game: '游戏',
      rogue: '肉鸽模式'
    },
    home: {
      title: '游戏自动化套件',
      status: '状态',
      start: '启动自动化',
      stop: '停止自动化',
      playerStats: '玩家状态',
      health: '生命值',
      mana: '魔法值',
      currentTask: '当前任务',
      progress: '进度'
    },
    game: {
      title: '游戏控制',
      playerStats: '玩家数据',
      health: '生命值',
      mana: '魔法值',
      inventory: '背包',
      currentTask: '当前任务',
      completeTask: '完成任务',
      battle: '战斗',
      startBattle: '开始战斗'
    },
    rogue: {
      title: '肉鸽模式',
      level: '第 {level} 层',
      artifacts: '神器',
      abilities: '能力',
      proceed: '前往下一层',
      endRun: '结束运行',
      selectCharacter: '选择角色',
      startRun: '开始运行',
      health: '生命值',
      attack: '攻击力',
      rewards: '奖励',
      continue: '继续'
    },
    about: {
      title: '关于',
      projectInfo: '项目信息',
      description: '游戏自动化套件是一个由AI驱动的高级游戏自动化和优化工具。',
      version: '版本',
      features: '功能特性',
      gameAutomation: '游戏自动化',
      gameAutomationDesc: '智能化的游戏任务和战斗自动化',
      aiDecision: 'AI决策',
      aiDecisionDesc: '先进的AI算法实现最优决策',
      rogueMode: '肉鸽模式',
      rogueModeDesc: '程序生成的挑战与独特奖励',
      dataAnalysis: '数据分析',
      dataAnalysisDesc: '全面的游戏性能和统计分析',
      techStack: '技术栈',
      contributors: '贡献者'
    }
  },
  'ja-JP': {
    general: {
      home: 'ホーム',
      about: '概要',
      game: 'ゲーム',
      rogue: 'ローグモード'
    },
    home: {
      title: 'ゲーム自動化スイート',
      status: 'ステータス',
      start: '自動化開始',
      stop: '自動化停止',
      playerStats: 'プレイヤーステータス',
      health: 'HP',
      mana: 'MP',
      currentTask: '現在のタスク',
      progress: '進捗'
    },
    game: {
      title: 'ゲームコントロール',
      playerStats: 'プレイヤー統計',
      health: 'HP',
      mana: 'MP',
      inventory: 'インベントリ',
      currentTask: '現在のタスク',
      completeTask: 'タスク完了',
      battle: 'バトル',
      startBattle: 'バトル開始'
    },
    rogue: {
      title: 'ローグモード',
      level: 'レベル {level}',
      artifacts: 'アーティファクト',
      abilities: 'アビリティ',
      proceed: '次のレベルへ',
      endRun: 'ラン終了',
      selectCharacter: 'キャラクター選択',
      startRun: 'ラン開始',
      health: 'HP',
      attack: '攻撃力',
      rewards: '報酬',
      continue: '続ける'
    },
    about: {
      title: '概要',
      projectInfo: 'プロジェクト情報',
      description: 'ゲーム自動化スイートは、AIを活用した高度なゲーム自動化・最適化ツールです。',
      version: 'バージョン',
      features: '機能',
      gameAutomation: 'ゲーム自動化',
      gameAutomationDesc: 'ゲームタスクとバトルのインテリジェントな自動化',
      aiDecision: 'AI意思決定',
      aiDecisionDesc: '最適な意思決定のための高度なAIアルゴリズム',
      rogueMode: 'ローグモード',
      rogueModeDesc: '手続き的に生成されるチャレンジとユニークな報酬',
      dataAnalysis: 'データ分析',
      dataAnalysisDesc: 'ゲームパフォーマンスと統計の包括的な分析',
      techStack: '技術スタック',
      contributors: 'コントリビューター'
    }
  }
}

export default createI18n({
  legacy: false,
  locale: 'zh-CN', // 默认语言
  fallbackLocale: 'en-US', // 回退语言
  messages
})

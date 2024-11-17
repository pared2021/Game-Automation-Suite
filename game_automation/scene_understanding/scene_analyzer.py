# 提取并存储模板特征
            self.template_features[template_name] = self._extract_features(image)
            
            detailed_logger.info(f"保存模板成功: {template_name}")
            return True
        except Exception as e:
            detailed_logger.error(f"保存模板失败: {str(e)}")
            return False

    @log_exception
    def get_scene_location(self, scene_name: str, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """获取场景在截图中的位置
        
        Args:
            scene_name: 场景名称
            screenshot: 截图数据

        Returns:
            Optional[Tuple[int, int]]: 场景位置坐标，未找到返回None
        """
        if scene_name not in self.templates:
            detailed_logger.warning(f"未找到模板: {scene_name}")
            return None
            
        try:
            template = self.templates[scene_name]
            
            # 转换为灰度图
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # 模板匹配
            result = cv2.matchTemplate(
                screenshot_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED
            )
            
            # 获取最佳匹配位置
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.8:  # 可配置的阈值
                # 返回模板中心点坐标
                h, w = template_gray.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                return (center_x, center_y)
            
            return None
        except Exception as e:
            detailed_logger.error(f"获取场景位置失败: {str(e)}")
            return None

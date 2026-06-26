/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.io.IOException
import java.util.{Collections, Map => JMap}

import com.codahale.metrics.MetricRegistry

import org.apache.spark.SparkConf
import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.resource.ResourceInformation
import org.scalatest.funsuite.AnyFunSuite

class GraphAutotuneRuntimeSuite extends AnyFunSuite {
  private val key = AutotuneStageKey(executionId = 11L, stageId = 3, stageAttemptId = 1)

  private class FailingPluginContext extends PluginContext {
    override def metricRegistry(): MetricRegistry = new MetricRegistry()

    override def conf(): SparkConf = new SparkConf(false)

    override def executorID(): String = "exec-1"

    override def hostname(): String = "localhost"

    override def resources(): JMap[String, ResourceInformation] = Collections.emptyMap()

    override def send(message: Any): Unit = throw new IOException("send failed")

    override def ask(message: Any): AnyRef = throw new IOException("ask failed")
  }

  test("stage runtime hint preserves cache key and expiration") {
    val hint = StageRuntimeHint(
      executionId = key.executionId,
      stageId = key.stageId,
      stageAttemptId = key.stageAttemptId,
      version = 7L,
      scan = ScanRuntimeHint(eagerPrefetch = true, minReadWindow = 1,
        maxReadWindow = 4, maxReadyBytes = 1024L),
      expiresAtNanos = 10L)

    assert(hint.key == key)
    assert(!hint.isExpired(9L))
    assert(hint.isExpired(10L))
  }

  test("hint cache memoizes no-hint responses by stage key") {
    var fetches = 0
    val cache = new AutotuneHintCache(stageKey => {
      assert(stageKey == key)
      fetches += 1
      AutotuneCachedHint.empty(stageKey)
    })

    val first = cache.get(key)
    val second = cache.get(key)

    assert(!first.hasHint)
    assert(first.version == 0L)
    assert(first eq second)
    assert(fetches == 1)
  }

  test("hint cache refreshes expired hints") {
    var version = 0L
    val cache = new AutotuneHintCache(stageKey => {
      version += 1
      AutotuneCachedHint(StageRuntimeHint(
        executionId = stageKey.executionId,
        stageId = stageKey.stageId,
        stageAttemptId = stageKey.stageAttemptId,
        version = version,
        scan = ScanRuntimeHint.empty,
        expiresAtNanos = 0L), hasHint = true)
    })

    assert(cache.get(key).version == 1L)
    assert(cache.get(key).version == 2L)
  }

  test("executor endpoint fails open when driver hint RPCs fail") {
    val conf = new RapidsConf(Map(RapidsConf.AUTOTUNE_FAIL_OPEN.key -> "true"))
    val endpoint = new RapidsAutotuneExecutorEndpoint(new FailingPluginContext, conf)

    val hint = endpoint.hintFor(key)
    assert(!hint.hasHint)
    assert(hint.version == 0L)
    endpoint.recordAppliedHint(key, taskAttemptId = 22L, partitionId = 1, hint)
  }
}

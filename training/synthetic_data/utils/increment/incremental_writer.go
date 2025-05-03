package increment

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

// Add counters to track messages
var (
	processedCount int32
	receivedCount  int32
	skippedCount   int32
)

// ModelResponse represents the full response from the model
type ModelResponse struct {
	Response string `json:"response"`
}

// EmailContent represents the actual email content we want to extract
type EmailContent struct {
	Text string `json:"text"`
}

// Message represents an individual message/result that you want to write to JSON.
type Message struct {
	ID       int    `json:"id"`
	Text     string `json:"text"`
	Attempts int    `json:"-"` // Track retry attempts
}

const (
	maxRetries = 3
	retryDelay = 500 * time.Millisecond
)

// WriteResultsIncrementally listens to the provided message channel,
// extracts just the email content, and writes it to the file.
func WriteResultsIncrementally(ctx context.Context, filePath string, msgChan <-chan Message, resultChan chan<- int, batchSize int) {
	log.Printf("Starting WriteResultsIncrementally for file: %s with batch size: %d", filePath, batchSize)
	
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("failed to open file for writing: %v", err)
	}
	defer f.Close()

	writer := bufio.NewWriter(f)
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)

	seenIDs := make(map[int]bool)
	retryQueue := make(chan Message, 100)
	
	// Create a batch buffer with dynamic size
	batch := make([]Message, 0, batchSize)

	// Helper function to flush the batch
	flushBatch := func() {
		log.Printf("Flushing batch of size %d", len(batch))
		for _, msg := range batch {
			if err := encoder.Encode(msg); err != nil {
				log.Printf("Failed to encode message %d: %v", msg.ID, err)
				continue
			}
			seenIDs[msg.ID] = true
			atomic.AddInt32(&processedCount, 1)
		}
		if err := writer.Flush(); err != nil {
			log.Printf("Error flushing writer: %v", err)
		}
		
		// Only send progress update after flushing batch
		count := atomic.LoadInt32(&processedCount)
		if count > 0 && count%10 == 0 {
			resultChan <- int(count)
		}
		
		log.Printf("Batch flush complete. Total processed: %d", count)
		
		// Clear the batch
		batch = batch[:0]
	}

	processMessageBatched := func(msg Message) {
		atomic.AddInt32(&receivedCount, 1)
		
		if seenIDs[msg.ID] {
			log.Printf("WARNING: Duplicate message ID received: %d", msg.ID)
			return
		}

		if msg.ID%10 == 0 {
			log.Printf("Processing message ID: %d (Received: %d, Processed: %d, Skipped: %d)", 
				msg.ID, 
				atomic.LoadInt32(&receivedCount),
				atomic.LoadInt32(&processedCount),
				atomic.LoadInt32(&skippedCount))
		}

		// Process message content (existing validation and cleaning logic)
		cleanText := processMessageContent(msg)
		if cleanText == "" {
			atomic.AddInt32(&skippedCount, 1)
			return
		}

		cleanMsg := Message{
			ID:   msg.ID,
			Text: cleanText,
		}

		// Add to batch
		batch = append(batch, cleanMsg)
		
		// Flush if batch is full - use the passed in batchSize parameter
		if len(batch) >= batchSize {
			flushBatch()
		}
	}

	// Add a timeout to flush batches that have been waiting too long
	const flushTimeout = 30 * time.Second
	flushTimer := time.NewTimer(flushTimeout)

	for {
		select {
		case <-flushTimer.C:
			if len(batch) > 0 {
				log.Printf("Flushing batch of size %d due to timeout of %d seconds", len(batch), int(flushTimeout.Seconds()))
				flushBatch()
			}
			flushTimer.Reset(flushTimeout)

		case <-ctx.Done():
			if len(batch) > 0 {
				flushBatch()
			}
			logStats()
			return

		case msg := <-retryQueue:
			if msg.Attempts >= maxRetries {
				log.Printf("Message %d: Failed after %d attempts", msg.ID, msg.Attempts)
				atomic.AddInt32(&skippedCount, 1)
				continue
			}
			time.Sleep(retryDelay)
			msg.Attempts++
			processMessageBatched(msg)

		case msg, ok := <-msgChan:
			if !ok {
				if len(batch) > 0 {
					flushBatch()
				}
				logStats()
				// Close the result channel to signal that all processing is done
				close(resultChan)
				return
			}
			processMessageBatched(msg)
		}
	}
}

// Helper function to process message content
func processMessageContent(msg Message) string {
	if msg.Text == "" {
		log.Printf("Message %d: Empty text received", msg.ID)
		return ""
	}

	if strings.Contains(msg.Text, "EOF") || strings.Contains(msg.Text, "\"error\":") {
        log.Printf("Message %d: Detected error response, skipping. Detail: %s", msg.ID, msg.Text)
        return ""
    }
	
	cleanText := msg.Text
	if len(cleanText) > 1000 {
		cleanText = cleanText[:1000]
	}

	// Try to parse as nested response
	var modelResp ModelResponse
	if err := json.Unmarshal([]byte(msg.Text), &modelResp); err == nil && modelResp.Response != "" {
		var emailContent EmailContent
		if err := json.Unmarshal([]byte(modelResp.Response), &emailContent); err == nil && emailContent.Text != "" {
			cleanText = emailContent.Text
		}
	}

	return strings.TrimSpace(cleanText)
}

func logStats() {
	log.Printf("Final stats - Received: %d, Processed: %d, Skipped: %d", 
		atomic.LoadInt32(&receivedCount),
		atomic.LoadInt32(&processedCount),
		atomic.LoadInt32(&skippedCount))
} 
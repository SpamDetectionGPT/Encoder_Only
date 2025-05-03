package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"syntheticgenerator/utils/increment"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

// worker generates synthetic content using langchaingo's Ollama integration
func worker(idx int, model string, prompt string, ctx context.Context, 
            client llms.LLM, ch chan<- increment.Message, 
            wg *sync.WaitGroup, sem chan struct{}) {
	defer wg.Done()
	defer func() { <-sem }() // Release semaphore when done
	
	fmt.Printf("Worker %d starting...\n", idx)
	
	// Add retry logic
	maxRetries := 3
	for retry := 0; retry <= maxRetries; retry++ {
		if retry > 0 {
			backoffTime := time.Duration(math.Pow(2, float64(retry-1))) * time.Second
			fmt.Printf("Worker %d: retry %d after %v delay\n", idx, retry, backoffTime)
			time.Sleep(backoffTime)
		}
		
		// Use langchaingo to generate response
		completion, err := llms.GenerateFromSinglePrompt(ctx, client, prompt, 
			llms.WithTemperature(0.8),
			llms.WithMaxTokens(512), // Increased to allow for longer, more detailed responses
			llms.WithStreamingFunc(nil), // Disable streaming
		)
		
		if err != nil {
			fmt.Printf("Worker %d: error generating content: %v\n", idx, err)
			continue // Retry if there's an error
		}
		
		// Clean up the output - remove any "Example" prefixes
		completion = removeExamplePrefix(completion)
		
		// Create JSON with the html tags properly preserved
		jsonResponse := map[string]interface{}{
			"text": completion,
		}
		
		// Use a custom JSON marshaler that doesn't escape HTML characters
		var buf bytes.Buffer
		encoder := json.NewEncoder(&buf)
		encoder.SetEscapeHTML(false) // This is the key to prevent HTML escaping
		if err := encoder.Encode(jsonResponse); err != nil {
			fmt.Printf("Worker %d: error marshaling response: %v\n", idx, err)
			ch <- increment.Message{ID: idx, Text: err.Error()}
			return
		}
		
		// Get the JSON string and trim newline that Encode adds
		responseText := strings.TrimSpace(buf.String())

		fmt.Printf("Worker %d completed.\n", idx)
		ch <- increment.Message{ID: idx, Text: responseText}
		
		// If successful, break out of retry loop
		return
	}
}

// removeExamplePrefix removes "Example X:" from the beginning of the text
func removeExamplePrefix(s string) string {
	// Remove any "Example X:" prefixes that may appear
	re := regexp.MustCompile(`(?i)^Example \d+:[\s]*`)
	return re.ReplaceAllString(s, "")
}

func main() {
	const numRequests = 50
	const maxConcurrentWorkers = 8
	const batchSize = 10
	const modelName = "syntheticgenerator" // Ollama model name
	
	// Initialize Ollama client with langchaingo
	client, err := ollama.New(
		ollama.WithServerURL("http://localhost:11434"),
		ollama.WithModel(modelName),
	)
	
	if err != nil {
		fmt.Printf("Error initializing Ollama client: %v\n", err)
		os.Exit(1)
	}
	
	prompt := "Generate a synthetic phishing email using the examples provided."
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// Increase buffer size to allow more pending messages
	msgChan := make(chan increment.Message, numRequests) 
	resultChan := make(chan int, 10) // Buffer for progress updates
	
	// Add file path constant
	const outputFile = "synthetic_emails_output.json"
	
	// Check if file is writable at startup
	f, err := os.OpenFile(outputFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Printf("Error checking output file access: %v\n", err)
		os.Exit(1)
	}
	f.Close()
	
	// Start the writer with explicit batch size
	go func() {
		increment.WriteResultsIncrementally(ctx, outputFile, msgChan, resultChan, batchSize)
		fmt.Println("Writer routine completed")
	}()
	
	// Start progress monitor
	go func() {
		for count := range resultChan {
			fmt.Printf("Progress: %d messages processed\n", count)
		}
	}()
	
	// Create a semaphore to limit concurrent workers
	sem := make(chan struct{}, maxConcurrentWorkers)
	
	// Create a WaitGroup to track workers
	var wg sync.WaitGroup
	wg.Add(numRequests)
	
	// Start workers with semaphore control
	fmt.Printf("Starting %d workers with max concurrency of %d and batch size of %d...\n", 
			   numRequests, maxConcurrentWorkers, batchSize)
	startTime := time.Now()
	
	for i := 0; i < numRequests; i++ {
		sem <- struct{}{} // Acquire semaphore (blocks when maxConcurrentWorkers reached)
		go worker(i, modelName, prompt, ctx, client, msgChan, &wg, sem)
	}
	
	// Wait for all workers to complete in a separate goroutine
	completionDone := make(chan struct{})
	go func() {
		wg.Wait()
		close(msgChan) // Signal that no more messages will be sent
		close(completionDone)
	}()
	
	// Wait for completion or handle other tasks
	<-completionDone
	elapsedTime := time.Since(startTime)
	
	// Wait for the result channel to be closed (indicating all processing is done)
	select {
	case <-time.After(5 * time.Second):
		fmt.Println("Timed out waiting for result channel to close")
	case _, ok := <-resultChan:
		if !ok {
			fmt.Println("Result channel closed naturally")
		}
	}
	
	// Verify file contents after completion
	if fileInfo, err := os.Stat(outputFile); err != nil {
		fmt.Printf("Error checking output file: %v\n", err)
	} else {
		fmt.Printf("Output file size: %d bytes\n", fileInfo.Size())
		fmt.Printf("Average processing time per request: %v\n", elapsedTime/time.Duration(numRequests))
	}
	
	fmt.Printf("All %d responses have been written to %s\n", numRequests, outputFile)
	fmt.Printf("Performance summary:\n")
	fmt.Printf("- Total time: %v\n", elapsedTime)
	fmt.Printf("- Requests: %d\n", numRequests)
	fmt.Printf("- Concurrency: %d\n", maxConcurrentWorkers)
	fmt.Printf("- Throughput: %.2f requests/second\n", float64(numRequests)/elapsedTime.Seconds())
	
	// Memory monitoring
	go func() {
		for {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			fmt.Printf("Memory usage: Alloc = %v MiB, TotalAlloc = %v MiB\n",
				m.Alloc/1024/1024, m.TotalAlloc/1024/1024)
			time.Sleep(5 * time.Second)
		}
	}()
}
